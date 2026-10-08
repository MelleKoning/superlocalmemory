import {env,createExecutionContext,waitOnExecutionContext} from 'cloudflare:test';import {expect,test,vi} from 'vitest';
import {generateKeyPair,exportJWK,SignJWT} from 'jose';
import {authFetch,type AuthWorkerEnv} from '../../src/worker-auth.ts';import {tokenHash} from '../../src/device-proof.ts';
import {enrolled,issuer} from './native-fixture.ts';

test('native PKCE bootstrap issues scoped owner token and idempotent encrypted device delivery',async()=>{
 const f=await enrolled();const first=await f.provision();expect(first.status).toBe(200);const device=await first.json() as {device_token:string};expect(device.device_token).toMatch(/^[a-f0-9]{64}$/);
 const second=await f.provision();expect(second.status).toBe(200);expect(await second.json()).toMatchObject(device);
 const owned=await env.OWNERS.getByName('16027584').getConnection('16027584',f.connectionId);expect(owned!.credentialEnvelope).not.toContain(device.device_token);
});

for(const route of ['bootstrap','oauth'])test(route+' cancellation fences a provision paused before owner insertion',async()=>{
 const f=await enrolled();const real=env.OWNERS.getByName('16027584');let release!:()=>void;let entered!:()=>void;
 const barrier=new Promise<void>(resolve=>release=resolve);const reached=new Promise<void>(resolve=>entered=resolve);
 f.configuration.OWNERS={getByName:()=>new Proxy(real,{get(target,key){if(key==='addConnection')return async(...args:Parameters<typeof real.addConnection>)=>{entered();await barrier;return real.addConnection(...args);};return (...args:unknown[])=>Reflect.get(target,key)(...args);}})} as unknown as typeof env.OWNERS;
 const provisioning=f.provision();await reached;
 try{
  const response=route==='bootstrap'?await f.call('/bootstrap/cancel',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({connection_id:f.connectionId,verifier:f.verifier})}):await f.call('/oauth/token',{method:'POST',headers:{'Content-Type':'application/x-www-form-urlencoded'},body:new URLSearchParams({token:f.issued.access_token,client_id:f.client.client_id})});
  expect(response.status).toBe(200);
 }finally{release();}
 expect((await provisioning).status).not.toBe(200);expect(await real.getConnection('16027584',f.connectionId)).toBeNull();
 expect((await env.RELAYS.getByName(f.connectionId).fetch(new Request('https://private.invalid/connector',{headers:{Upgrade:'websocket',Authorization:'Bearer '+'a'.repeat(64)}}))).status).not.toBe(101);
});

test('native OAuth revocation closes an established relay and denies an existing MCP authorization',async()=>{
 const {resourceGateway}=await import('../../src/worker-resource.ts');const {decodeRelayFrame,encodeRelayFrame}=await import('../../src/relay-protocol.ts');
 const f=await enrolled();const delivered=await f.provision();expect(delivered.status).toBe(200);const device=await delivered.json() as {device_token:string};
 const relay=env.RELAYS.getByName(f.connectionId);const registry=env.REGISTRIES.getByName(f.connectionId);const audience='https://mcp.superlocalmemory.com/mcp';
 await registry.addAuthorization({ownerId:'16027584',connectionId:f.connectionId,authorizationId:'existing-mcp',clientId:'synthetic-mcp',audience,consentedTools:['recall'],consentedScopes:['slm:read'],consentedCorrection:false,consentedSharedRead:false,consentedGlobalRead:false,authorizationVersion:1,revokedAt:null});
 const configuration={...env,AUTH_SERVER:{async validateToken(){return {userId:'16027584',clientId:'synthetic-mcp',props:{ownerId:'16027584',connectionId:f.connectionId,authorizationId:'existing-mcp'},scope:['slm:read'],expiresAt:Math.floor(Date.now()/1000)+60,audience};}}};
 const upgrade=await relay.fetch(new Request('https://private.invalid/connector',{headers:{Upgrade:'websocket',Authorization:'Bearer '+device.device_token}}));expect(upgrade.status).toBe(101);const socket=upgrade.webSocket!;
 const ready=new Promise<void>(resolve=>socket.addEventListener('message',()=>resolve(),{once:true}));socket.accept();await ready;
 socket.addEventListener('message',event=>{const parsed=decodeRelayFrame(String(event.data));if(!parsed.ok||parsed.frame.kind!=='request')return;const frame=parsed.frame;const rpc=JSON.parse(atob(frame.bodyBase64));const encoded=encodeRelayFrame({v:1,kind:'response',id:frame.id,generation:frame.generation,status:200,headers:[['content-type','application/json']],bodyBase64:btoa(JSON.stringify({jsonrpc:'2.0',id:rpc.id,result:{content:[{type:'text',text:'synthetic memory'}]}}))});if(encoded.ok)socket.send(encoded.text);});
 async function recall(){const context=createExecutionContext();const response=await resourceGateway.fetch(new Request(audience,{method:'POST',headers:{Authorization:'Bearer synthetic-mcp','Content-Type':'application/json',Accept:'application/json'},body:JSON.stringify({jsonrpc:'2.0',id:1,method:'tools/call',params:{name:'recall',arguments:{query:'synthetic'}}})}),configuration as never,context);await waitOnExecutionContext(context);return response;}
 try{expect((await recall()).status).toBe(200);const closed=new Promise<void>(resolve=>socket.addEventListener('close',()=>resolve(),{once:true}));
  const revoked=await f.call('/oauth/token',{method:'POST',headers:{'Content-Type':'application/x-www-form-urlencoded'},body:new URLSearchParams({token:f.issued.access_token,client_id:f.client.client_id})});expect(revoked.status).toBe(200);await closed;
  expect((await recall()).status).toBe(403);expect((await relay.fetch(new Request('https://private.invalid/connector',{headers:{Upgrade:'websocket',Authorization:'Bearer '+device.device_token}}))).status).toBe(403);
 }finally{socket.close();}
});

async function connectedApp(f:Awaited<ReturnType<typeof enrolled>>,clientName:string,authorizationId:string){
 const registered=await f.call('/oauth/register',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({client_name:clientName,redirect_uris:['https://backend.composio.dev/api/v1/auth-apps/add'],token_endpoint_auth_method:'none'})});
 const client=await registered.json() as {client_id:string};
 await env.REGISTRIES.getByName(f.connectionId).addAuthorization({ownerId:'16027584',connectionId:f.connectionId,authorizationId,clientId:client.client_id,audience:'https://mcp.superlocalmemory.com/mcp',consentedTools:['recall','search','fetch','get_status','remember'],consentedScopes:['slm:read','slm:write'],consentedCorrection:false,consentedSharedRead:false,consentedGlobalRead:false,authorizationVersion:1,revokedAt:null});
 return client.client_id;
}

test('owner lists connected apps by name and removing one cuts access at once',async()=>{
 const f=await enrolled();expect((await f.provision()).status).toBe(200);
 const clientId=await connectedApp(f,'Composio','app-composio');
 const listed=await f.ownerCall('/owner/apps');expect(listed.status).toBe(200);
 const body=await listed.json() as {apps:Array<Record<string,unknown>>};
 expect(body.apps).toHaveLength(1);
 expect(body.apps[0]).toMatchObject({authorization_id:'app-composio',name:'Composio',client_host:'backend.composio.dev',permissions:{read:true,save:true,session:false},version:1,last_used_at_ms:null});
 expect(typeof body.apps[0].connected_at_ms).toBe('number');
 const removed=await f.ownerCall('/owner/apps/revoke',{authorization_id:'app-composio',expected_version:1});
 expect(removed.status).toBe(200);expect(await removed.json()).toEqual({revoked:true,version:2});
 const actor={ownerId:'16027584',connectionId:f.connectionId,authorizationId:'app-composio',clientId,audience:'https://mcp.superlocalmemory.com/mcp',credentialKind:'oauth' as const,scopes:['slm:read' as const]};
 expect(await env.REGISTRIES.getByName(f.connectionId).admit(actor,'https://mcp.superlocalmemory.com/mcp',{era:'legacy',rpcMethod:'tools/call',toolName:'recall',arguments:{query:'x'},originalBody:new Uint8Array()})).toMatchObject({allowed:false});
 expect((await (await f.ownerCall('/owner/apps')).json() as {apps:unknown[]}).apps).toEqual([]);
});

test('removing an app needs a well-formed request and the current version',async()=>{
 const f=await enrolled();expect((await f.provision()).status).toBe(200);await connectedApp(f,'Muse','app-muse');
 expect((await f.ownerCall('/owner/apps/revoke',{authorization_id:'app-muse',expected_version:7})).status).toBe(409);
 expect((await f.ownerCall('/owner/apps/revoke',{authorization_id:'app-muse'})).status).toBe(400);
 expect((await f.ownerCall('/owner/apps/revoke',{authorization_id:'../../etc',expected_version:1})).status).toBe(400);
 expect((await f.ownerCall('/owner/apps/revoke',{authorization_id:'not-there',expected_version:1})).status).toBe(404);
});

test('third-party app names come back as bounded plain text',async()=>{
 const f=await enrolled();expect((await f.provision()).status).toBe(200);
 await connectedApp(f,'Evil\u0000\u001b[31m'+'x'.repeat(500),'app-hostile');
 const body=await (await f.ownerCall('/owner/apps')).json() as {apps:Array<{name:string}>};
 expect(body.apps[0].name.length).toBeLessThanOrEqual(80);expect(body.apps[0].name).not.toMatch(/[\x00-\x1f\x7f]/);expect(body.apps[0].name.startsWith('Evil')).toBe(true);
});

test('a Connected apps outage is reported as unavailable, not as a sign-in failure',async()=>{
 const f=await enrolled();expect((await f.provision()).status).toBe(200);const real=env.REGISTRIES;
 f.configuration.REGISTRIES={getByName:(name:string)=>new Proxy(real.getByName(name),{get(target,key){if(key==='listAuthorizations')return async()=>{throw new Error('synthetic_outage');};return (...args:unknown[])=>Reflect.get(target,key)(...args);}})} as unknown as typeof env.REGISTRIES;
 const listed=await f.ownerCall('/owner/apps');
 expect(listed.status).toBe(503);expect(await listed.json()).toEqual({error:'owner_operation_unavailable'});
});
