/** A connected laptop renews its own credential before it expires, and grants that keep
 * being used never expire. Without this every Web access link and every connected app
 * stopped 30 days after sign-in. */
import {env} from 'cloudflare:test';import {afterEach,expect,test,vi} from 'vitest';
import {enrolled} from './native-fixture.ts';
import {DEVICE_CREDENTIAL_TTL_MS,RENEWAL_WINDOW_MS} from '../../src/credential-lifetime.ts';
const DAY=24*3600*1000;
const OWNER='16027584';
type Delivery={device_token:string;expires_at_ms:number;generation:number};
afterEach(()=>{vi.useRealTimers();});
function at(ms:number){vi.useFakeTimers({toFake:['Date']});vi.setSystemTime(ms);}
/** Move the clock and refresh the owner sign-in, as the laptop does before renewing. */
async function later(f:Awaited<ReturnType<typeof enrolled>>,ms:number){at(ms);const refreshed=await f.refreshOwner();expect(refreshed.status).toBe(200);}
async function relayAccepts(connectionId:string,token:string):Promise<number>{
 const response=await env.RELAYS.getByName(connectionId).fetch(new Request('https://private.invalid/connector',{headers:{Upgrade:'websocket',Authorization:'Bearer '+token}}));
 response.webSocket?.accept();response.webSocket?.close();return response.status;
}
async function provisioned(){
 const f=await enrolled();const response=await f.provision();expect(response.status).toBe(200);
 return {f,first:await response.json() as Delivery};
}

test('the credential lasts 30 days and renewal opens in its second half',()=>{
 expect(DEVICE_CREDENTIAL_TTL_MS).toBe(30*DAY);expect(RENEWAL_WINDOW_MS).toBe(15*DAY);
});

test('renewal is refused while more than half of the credential life remains',async()=>{
 const {f,first}=await provisioned();
 const early=await f.ownerCall('/owner/renew',{expected_generation:first.generation});
 expect(early.status).toBe(409);expect(await early.json()).toEqual({error:'renewal_not_due'});
});

test('renewal in the second half issues a new credential and retires the old one',async()=>{
 const {f,first}=await provisioned();const start=Date.now();
 await later(f,start+16*DAY);
 const renewed=await f.ownerCall('/owner/renew',{expected_generation:first.generation});expect(renewed.status).toBe(200);
 const next=await renewed.json() as Delivery&{connection_id:string};
 expect(next.generation).toBe(first.generation+1);expect(next.device_token).toMatch(/^[a-f0-9]{64}$/);expect(next.device_token).not.toBe(first.device_token);
 expect(next.expires_at_ms).toBe(start+16*DAY+DEVICE_CREDENTIAL_TTL_MS);expect(next.connection_id).toBe(f.connectionId);
 vi.useRealTimers();
 expect(await relayAccepts(f.connectionId,first.device_token)).toBe(401);
 expect(await relayAccepts(f.connectionId,next.device_token)).toBe(101);
 const owned=await env.OWNERS.getByName(OWNER).getConnection(OWNER,f.connectionId);
 expect(owned!.generation).toBe(next.generation);expect(owned!.deviceExpiresAtMs).toBe(next.expires_at_ms);expect(owned!.credentialEnvelope).not.toContain(next.device_token);
 expect(await env.DEVICES.getByName(owned!.deviceDigest).lookup(owned!.deviceDigest)).not.toBeNull();
 const fetched=await f.provision();expect(fetched.status).toBe(200);expect(await fetched.json()).toMatchObject({device_token:next.device_token,generation:next.generation});
});

test('a connected app keeps working past the first credential\'s expiry once renewed',async()=>{
 const {f,first}=await provisioned();const start=Date.now();const audience='https://mcp.superlocalmemory.com/mcp';
 const registry=env.REGISTRIES.getByName(f.connectionId);
 await registry.addAuthorization({ownerId:OWNER,connectionId:f.connectionId,authorizationId:'app-a',clientId:'synthetic-app',audience,consentedTools:['recall'],consentedScopes:['slm:read'],consentedCorrection:false,consentedSharedRead:false,consentedGlobalRead:false,authorizationVersion:1,revokedAt:null});
 const admit=()=>registry.admit({ownerId:OWNER,connectionId:f.connectionId,authorizationId:'app-a',clientId:'synthetic-app',audience,credentialKind:'oauth',scopes:['slm:read']},audience,{era:'legacy',rpcMethod:'initialize',rpcId:1,originalBody:new Uint8Array()});
 at(start+31*DAY);expect((await admit()).code).toBe('ENTITLEMENT_REQUIRED');
 await later(f,start+20*DAY);const renewed=await f.ownerCall('/owner/renew',{expected_generation:first.generation});expect(renewed.status).toBe(200);
 at(start+40*DAY);expect((await admit()).allowed).toBe(true);
});

test('two renewals at once produce exactly one new credential',async()=>{
 const {f,first}=await provisioned();await later(f,Date.now()+16*DAY);
 const results=await Promise.all([f.ownerCall('/owner/renew',{expected_generation:first.generation}),f.ownerCall('/owner/renew',{expected_generation:first.generation})]);
 expect(results.map(r=>r.status).sort()).toEqual([200,409]);
 const owned=await env.OWNERS.getByName(OWNER).getConnection(OWNER,f.connectionId);expect(owned!.generation).toBe(first.generation+1);
});

test('a stale generation is refused and the laptop can fetch the current credential',async()=>{
 const {f,first}=await provisioned();await later(f,Date.now()+16*DAY);
 const renewed=await f.ownerCall('/owner/renew',{expected_generation:first.generation});expect(renewed.status).toBe(200);const next=await renewed.json() as Delivery;
 const stale=await f.ownerCall('/owner/renew',{expected_generation:first.generation});expect(stale.status).toBe(409);expect(await stale.json()).toEqual({error:'version_conflict'});
 expect(await (await f.provision()).json()).toMatchObject({device_token:next.device_token});
});

test('a renewal request must be exactly one positive generation',async()=>{
 const {f,first}=await provisioned();await later(f,Date.now()+16*DAY);
 for(const body of [undefined,{},{expected_generation:0},{expected_generation:'1'},{expected_generation:first.generation,extra:true}])
  expect((await f.ownerCall('/owner/renew',body)).status).toBe(400);
});

test('a revoked connection cannot renew, and its sign-in can no longer be refreshed',async()=>{
 const {f,first}=await provisioned();
 expect((await f.ownerCall('/owner/revoke')).status).toBe(200);
 expect((await f.ownerCall('/owner/renew',{expected_generation:first.generation})).status).toBe(403);
 at(Date.now()+16*DAY);
 expect((await f.refreshOwner()).status).toBe(400);
 expect((await f.ownerCall('/owner/renew',{expected_generation:first.generation})).status).toBe(401);
});

test('an open relay on the old credential is closed as rotated',async()=>{
 const {f,first}=await provisioned();
 const upgrade=await env.RELAYS.getByName(f.connectionId).fetch(new Request('https://private.invalid/connector',{headers:{Upgrade:'websocket',Authorization:'Bearer '+first.device_token}}));
 expect(upgrade.status).toBe(101);const socket=upgrade.webSocket!;const closed=new Promise<void>(resolve=>socket.addEventListener('close',()=>resolve(),{once:true}));socket.accept();
 await later(f,Date.now()+16*DAY);expect((await f.ownerCall('/owner/renew',{expected_generation:first.generation})).status).toBe(200);
 await closed;
});

test('a renewal interrupted after the owner record changed is completed by provisioning',async()=>{
 const {f,first}=await provisioned();const real=env.RELAYS;let failOnce=true;
 f.configuration.RELAYS={getByName:(name:string)=>new Proxy(real.getByName(name),{get(target,key){if(key==='configureBinding'&&failOnce){failOnce=false;return async()=>{throw new Error('synthetic_outage');};}return (...args:unknown[])=>Reflect.get(target,key)(...args);}})} as unknown as typeof env.RELAYS;
 await later(f,Date.now()+16*DAY);
 const interrupted=await f.ownerCall('/owner/renew',{expected_generation:first.generation});expect(interrupted.status).toBe(503);expect(await interrupted.json()).toEqual({error:'owner_operation_unavailable'});
 const repaired=await f.provision();expect(repaired.status).toBe(200);const current=await repaired.json() as Delivery;
 expect(current.generation).toBe(first.generation+1);
 vi.useRealTimers();
 expect(await relayAccepts(f.connectionId,current.device_token)).toBe(101);
 expect(await relayAccepts(f.connectionId,first.device_token)).toBe(401);
});

test('an owner sign-in that keeps refreshing never expires, and one left idle for 30 days does',async()=>{
 const {f}=await provisioned();const start=Date.now();
 for(const day of [25,50,75]){at(start+day*DAY);expect((await f.refreshOwner()).status).toBe(200);}
 at(start+(75+31)*DAY);expect((await f.refreshOwner()).status).toBe(400);
});
