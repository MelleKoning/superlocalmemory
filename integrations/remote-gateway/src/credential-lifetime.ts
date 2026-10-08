/** How long a laptop's gateway credential lives, and when it may be renewed.
 * The laptop renews in the second half of the credential's life; the gateway
 * refuses earlier renewals, so a credential is replaced at most about twice a month. */
export const DEVICE_CREDENTIAL_TTL_MS=30*24*3600*1000;
export const RENEWAL_WINDOW_MS=DEVICE_CREDENTIAL_TTL_MS/2;
/** A sign-in grant (the owner's, or a connected app's) slides forward this long on
 * every refresh and expires after this long unused. The OAuth provider and the
 * issued-token ledger both use it, so they can never disagree. */
export const GRANT_IDLE_TTL_S=30*24*3600;
