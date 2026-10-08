/** One real native enrollment through the auth Worker: DCR, PKCE, bootstrap, consent,
 * GitHub sign-in (stubbed upstream), owner token, plus helpers for owner operations. */
import {env,createExecutionContext,waitOnExecutionContext} from 'cloudflare:test';import {expect,vi} from 'vitest';
import {generateKeyPair,exportJWK,SignJWT} from 'jose';
import {authFetch,type AuthWorkerEnv} from '../../src/worker-auth.ts';import {tokenHash} from '../../src/device-proof.ts';
export const issuer='https://auth.superlocalmemory.com';
export const cookies=(response:Response)=>response.headers.getSetCookie().map(x=>x.split(';')[0]).join('; ');
function handle(page:string){return /name="handle" value="([^"]+)"/.exec(page)![1];}
export async function enrolled(){
 const configuration={...env,GITHUB_CLIENT_ID:'synthetic-client',GITHUB_CLIENT_SECRET:'synthetic-secret',DEVICE_WRAP_KEY:'a'.repeat(64)} as AuthWorkerEnv;
 const pair=await generateKeyPair('ES256',{extractable:true});const jwk=await exportJWK(pair.publicKey);const connectionId=crypto.randomUUID().replaceAll('-','');const profileId='synthetic';const installationId='i-'+connectionId;const verifier='v'.repeat(43);const redirect='http://127.0.0.1:18767/api/v3/connections/callback';
 async function call(path:string,init?:RequestInit){const ctx=createExecutionContext();const response=await authFetch(new Request(issuer+path,init),configuration,ctx);await waitOnExecutionContext(ctx);return response;}
 const registration=await call('/oauth/register',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({client_name:'SuperLocalMemory Desktop',redirect_uris:[redirect],token_endpoint_auth_method:'none',grant_types:['authorization_code','refresh_token'],response_types:['code']})});expect(registration.status).toBe(201);const client=await registration.json() as {client_id:string};
 const params=new URLSearchParams({response_type:'code',client_id:client.client_id,redirect_uri:redirect,scope:'slm:connect',state:'s'.repeat(43),code_challenge:await tokenHash(verifier),code_challenge_method:'S256',resource:issuer+'/owner'});
 const started=await call('/bootstrap',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({connectionId,installationId,profileId,host:'muse',permissions:{read:true,write:true,correction:false,session:false},deviceJwk:jwk,expiresAtMs:Date.now()+600000,authorizationUrl:issuer+'/authorize?'+params})});expect(started.status).toBe(201);
 const consent=await call('/owner-login?connection_id='+connectionId);expect(consent.status).toBe(200);
 const signIn=await call('/consent',{method:'POST',headers:{Origin:issuer,Cookie:cookies(consent),'Content-Type':'application/x-www-form-urlencoded'},body:new URLSearchParams({handle:handle(await consent.text()),decision:'allow'})});expect(signIn.status).toBe(302);
 const upstream=new URL(signIn.headers.get('Location')!);
 vi.stubGlobal('fetch',async(input:RequestInfo|URL)=>{if(String(input)==='https://github.com/login/oauth/access_token')return Response.json({access_token:'synthetic-provider-token'});if(String(input)==='https://api.github.com/user')return Response.json({id:16027584});throw new Error('unexpected outbound request');});
 try{
  const returned=await call('/github/callback?'+new URLSearchParams({code:'synthetic-code',state:upstream.searchParams.get('state')!}),{headers:{Cookie:cookies(signIn)}});expect(returned.status).toBe(302);const callback=new URL(returned.headers.get('Location')!);
  const tokenResponse=await call('/oauth/token',{method:'POST',headers:{'Content-Type':'application/x-www-form-urlencoded'},body:new URLSearchParams({grant_type:'authorization_code',client_id:client.client_id,redirect_uri:redirect,code:callback.searchParams.get('code')!,code_verifier:verifier,resource:issuer+'/owner'})});expect(tokenResponse.status).toBe(200);const issued=await tokenResponse.json() as {access_token:string;refresh_token:string;scope:string};expect(issued.scope).toBe('slm:connect');
  async function provision(){const proof=await new SignJWT({htm:'POST',htu:issuer+'/owner/connections',ath:await tokenHash(issued.access_token)}).setProtectedHeader({typ:'dpop+jwt',alg:'ES256',jwk}).setIssuedAt().setJti(crypto.randomUUID()).sign(pair.privateKey);return call('/owner/connections',{method:'POST',headers:{Authorization:'Bearer '+issued.access_token,DPoP:proof}});}
  async function ownerCall(path:string,body?:unknown){const proof=await new SignJWT({htm:'POST',htu:issuer+path,ath:await tokenHash(issued.access_token)}).setProtectedHeader({typ:'dpop+jwt',alg:'ES256',jwk}).setIssuedAt().setJti(crypto.randomUUID()).sign(pair.privateKey);return call(path,{method:'POST',headers:{Authorization:'Bearer '+issued.access_token,DPoP:proof,...(body===undefined?{}:{'Content-Type':'application/json'})},...(body===undefined?{}:{body:JSON.stringify(body)})});}
  /** What the laptop does before an owner operation once its hour-long access token has lapsed. */
  async function refreshOwner(){const response=await call('/oauth/token',{method:'POST',headers:{'Content-Type':'application/x-www-form-urlencoded'},body:new URLSearchParams({grant_type:'refresh_token',refresh_token:issued.refresh_token,client_id:client.client_id,resource:issuer+'/owner'})});if(response.status===200){const next=await response.clone().json() as {access_token:string;refresh_token:string};issued.access_token=next.access_token;issued.refresh_token=next.refresh_token;}return response;}
  return {configuration,connectionId,profileId,installationId,verifier,client,issued,call,provision,ownerCall,refreshOwner,pair,jwk};
 }finally{vi.unstubAllGlobals();}
}
