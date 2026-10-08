/** A laptop that has gone quiet (asleep, or its network dropped without a close) is
 * reported at once instead of every call waiting out the full relay budget. */
import { env } from 'cloudflare:workers';
import { afterEach, expect, test, vi } from 'vitest';
import type { RelayDO } from '../../src/relay-do.ts';
import { CONNECTOR_SILENCE_MS } from '../../src/relay-do.ts';
import type { RelayFrame } from '../../src/relay-protocol.ts';
import { decodeRelayFrame, encodeRelayFrame } from '../../src/relay-protocol.ts';
const sockets: WebSocket[] = [];
afterEach(() => { vi.useRealTimers(); for (const ws of sockets.splice(0)) { try { ws.close(); } catch {} } });
async function setup() {
  const token=crypto.randomUUID().replaceAll('-','')+crypto.randomUUID().replaceAll('-','');
  const digest=Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',new TextEncoder().encode(token))),x=>x.toString(16).padStart(2,'0')).join('');
  const stub=env.RELAYS.getByName(crypto.randomUUID());
  await stub.configureBinding({ownerId:'owner-a',connectionId:'connection-a',installationId:'installation-a',profileId:'profile-a',deviceDigest:digest,deviceExpiresAt:Date.now()+3600000});
  return {stub,token};
}
function nextMessage(ws:WebSocket):Promise<string> { return new Promise(resolve=>ws.addEventListener('message',e=>resolve(String(e.data)),{once:true})); }
async function connect(stub:DurableObjectStub<RelayDO>,token:string) {
  const response=await stub.fetch(new Request('https://private.invalid/connector',{headers:{Upgrade:'websocket',Authorization:'Bearer '+token}}));
  expect(response.status).toBe(101);const ws=response.webSocket!;sockets.push(ws);const ready=nextMessage(ws);ws.accept();
  return {ws,generation:(JSON.parse(await ready) as {generation:number}).generation};
}
function frame(generation:number) { return {v:1 as const,kind:'request' as const,id:crypto.randomUUID(),generation,deadlineAt:Date.now()+20000,headers:[['Content-Type','application/json']] as [string,string][],bodyBase64:btoa('{"jsonrpc":"2.0","id":1,"method":"tools/list"}')}; }
function answer(ws:WebSocket,text:string) {
  const d=decodeRelayFrame(text);if(!d.ok||d.frame.kind!=='request')throw new Error('expected request');const request=d.frame as Extract<RelayFrame,{kind:'request'}>;
  const e=encodeRelayFrame({v:1,kind:'response',id:request.id,generation:request.generation,status:200,headers:[['Content-Type','application/json']],bodyBase64:btoa('{}')});if(e.ok)ws.send(e.text);
}
function later(ms:number){vi.useFakeTimers({toFake:['Date']});vi.setSystemTime(Date.now()+ms);}

test('the silence limit allows two missed heartbeats',()=>{expect(CONNECTOR_SILENCE_MS).toBe(45000);});

test('a laptop silent past the limit is reported asleep at once, not after the full budget',async()=>{
  const {stub,token}=await setup();const {generation}=await connect(stub,token);
  later(CONNECTOR_SILENCE_MS+15000);const started=performance.now();
  const response=await stub.forward(frame(generation));
  expect(response.status).toBe(503);expect(await response.json()).toEqual({error:'connector_asleep'});
  expect(performance.now()-started).toBeLessThan(2000);
});

test('a heartbeat keeps a quiet link usable',async()=>{
  const {stub,token}=await setup();const {ws,generation}=await connect(stub,token);
  const pong=nextMessage(ws);ws.send('ping');expect(await pong).toBe('pong');
  later(30000);const delivered=nextMessage(ws);const result=stub.forward(frame(generation));answer(ws,await delivered);
  expect((await result).status).toBe(200);
});

test('a link that has just connected counts as heard',async()=>{
  const {stub,token}=await setup();const {ws,generation}=await connect(stub,token);
  later(20000);const delivered=nextMessage(ws);const result=stub.forward(frame(generation));answer(ws,await delivered);
  expect((await result).status).toBe(200);
});

test('a laptop that answers requests is heard even without heartbeats',async()=>{
  const {stub,token}=await setup();const {ws,generation}=await connect(stub,token);
  later(30000);let delivered=nextMessage(ws);let result=stub.forward(frame(generation));answer(ws,await delivered);expect((await result).status).toBe(200);
  later(30000);delivered=nextMessage(ws);result=stub.forward(frame(generation));answer(ws,await delivered);expect((await result).status).toBe(200);
});
