+++
date = '2026-03-01T11:00:36+02:00'
draft = true
title = 'Why Hashing Passwords Is not Enough - and How Zero-Knowledge Authentication Fixes It'
tags = ['security', 'authentication', 'cryptography']
math = true
+++

Every few months, another company makes the news for leaking its users' passwords. The post-mortem is always roughly the same: engineers reassure everyone that passwords were *hashed and salted*, so users are probably fine. And yet, passwords still end up in plaintext in the wild. How?

The answer is that hashing protects against one specific attacker — but not the one that is often already inside your system.

## The Threat Model People Miss

When we talk about hashing passwords, we implicitly assume the attacker's scenario: someone steals a database dump and tries to crack it offline. A strong hash with a unique salt per user makes that extremely slow. It's good protection against that specific threat.

But there is another attacker: the **persistent server-side attacker**. This could be a compromised server, a rogue employee with production access, or an intruder who has quietly persisted inside your infrastructure. This attacker doesn't need to crack a database dump. They can simply wait.

On every login, a user's plaintext password travels from the browser to your server. It arrives, it gets hashed, the hash is compared. That window — however brief — is an opportunity. A piece of logging middleware, an in-memory inspection tool, or a patched binary can intercept the plaintext before hashing ever runs. No amount of bcrypt helps you here.

## How Password Hashing Works (and Where It Falls Short)

To be precise about the gap, here is what a typical login flow looks like:

1. User types password in the browser.
2. Browser sends it to the server (over TLS, so it's encrypted in transit).
3. Server receives the **plaintext** password.
4. Server hashes it with the stored salt and compares to the stored hash.
5. Match → session granted.

Step 3 is the problem. The server sees the plaintext. Every time. TLS protects against network eavesdroppers, but once the data is on your server, TLS has done its job and stepped aside.

Salting and hashing protect a *stolen database*. They do nothing to protect against anyone with code execution on the server handling step 3.

## Zero-Knowledge Authentication — The Concept

Zero-knowledge authentication flips the model: the server **never learns the password**, not even for an instant. Instead, the client mathematically proves to the server that it knows the password, without sending it — or anything derived from it that could be replayed.

The intuition: imagine proving to someone you know the combination to a safe by opening it in front of them, without ever saying the combination out loud. They are convinced you know it. They cannot repeat what you did, because you never gave them the information — only the proof.

In cryptographic terms, the two parties engage in an interactive protocol. At the end, the server is convinced of the user's identity, and the observed transcript of the exchange is useless to an attacker who recorded it.

## A Concrete Protocol: Schnorr-Based Authentication
{{< callout type="warning" >}}
**Note:** What follows is a textbook construction for illustration purposes. It is not production-ready — the caveats at the end of this section explain why.
{{< /callout >}}
To make zero-knowledge authentication concrete, we can build a simple password authentication scheme on top of the **Schnorr identification protocol**. The security of the construction rests on a standard cryptographic hardness assumption: the **Decisional Diffie-Hellman (DDH) assumption**.

### The DDH Assumption
We work in a cyclic group $\mathbb{G}$ of prime order $q$ with a publicly known generator $g$. The discrete logarithm problem says: given $g^x$, it is computationally infeasible to recover $x$. DDH goes further. 

{{< definition title="DDH Assumption" >}}
Given a tuple $(g,\, g^a,\, g^b,\, g^c)$ where $a$ and $b$ are chosen at random, it is computationally infeasible to distinguish whether $c = ab \pmod{q}$ (a "DH tuple") or $c$ is an independent random value. In other words, $g^{ab}$ looks random to an efficient observer who only sees $g^a$ and $g^b$.
{{< /definition >}}

This assumption underlies the security of Diffie-Hellman key exchange, ElGamal encryption, and — as we will see — the zero-knowledge property of the Schnorr protocol.

### Registration (one time)

1. The client derives a private key from the password: $x = H(\texttt{password}) \pmod{q}$, where $H$ is a cryptographic hash function.
2. The client computes the corresponding public key: $X = g^x \pmod{p}$.
3. The client sends `(username, X)` to the server. The server stores `(username, X)`. **The password never leaves the client.**

The server now holds a public key, not a password or a hash of one.

### Login

The login is a three-message **Sigma protocol** (also called a $\Sigma$-protocol or commit-challenge-respond protocol):

1. **Commit.** The client picks a uniformly random $r \leftarrow \mathbb{Z}_q$, computes the commitment $R = g^r \pmod{p}$, and sends `(username, R)` to the server.
2. **Challenge.** The server picks a uniformly random challenge $c \leftarrow \mathbb{Z}_q$ and sends it to the client.
3. **Respond.** The client computes the response $s = r + x \cdot c \pmod{q}$ and sends $s$ to the server.
4. **Verify.** The server checks whether $g^s \equiv R \cdot X^c \pmod{p}$.

If the check passes, the server is convinced the client knows $x$ — and therefore the password. If the client does not know $x$, they cannot produce a valid $s$ for a fresh random challenge (except with negligible probability).

### Why the Server Learns Nothing

The transcript $(R, c, s)$ can be *simulated* without knowing $x$: pick any random $s$ and $c$, then set $R = g^s \cdot X^{-c} \pmod{p}$. The resulting tuple is indistinguishable from a real transcript under DDH. This is the formal definition of **honest-verifier zero-knowledge**: the server's view during login contains no information about $x$ beyond the fact that the client knows it.

### Caveats: Why This Is Textbook Only

This construction illustrates the core idea cleanly, but several properties of a production protocol are missing:

- **Offline dictionary attack on the public key.** The server stores $X = g^{H(\texttt{password})}$. An attacker who steals the database can guess passwords offline and check each guess against $X$ — a significant weakness that real protocols like SRP and OPAQUE specifically address.
- **No password hardening.** Computing $x = H(\texttt{password})$ is fast. A real scheme would use a memory-hard function (Argon2, scrypt) to slow down guessing.
- **No mutual authentication.** The client proves its identity to the server, but the server does not prove its identity to the client beyond the TLS certificate.
- **Malleability and replay.** Without careful session binding and nonce management, variants of this protocol can be vulnerable to replay and man-in-the-middle attacks.
- **Non-standard group parameters.** Choosing safe $p$ and $q$ correctly is non-trivial; use well-audited parameters (e.g., RFC 3526 groups).

At no point during the login does the server receive the password or anything it could use to reconstruct it. That core guarantee holds. But the gap between this sketch and a deployable protocol is where most of the engineering lives.

## What This Protects Against (and What It Doesn't)

**Threats mitigated:**
- A persistent attacker with code execution on the server cannot intercept a password — there is no password to intercept.
- A stolen database gives an attacker the verifier, not the password. Cracking the verifier is computationally equivalent to cracking the original password hash, and the verifier cannot be replayed.
- Logging bugs, overly verbose middleware, crash dumps — none of these leak a password that was never there.

**Remaining threats:**
- **Client-side malware** can read the password from the input field before the protocol runs. ZK auth does not help here.
- **Phishing** is unaffected — a user typing their password into a fake login page still hands it to the attacker.
- **Implementation bugs** in the SRP library can break the security guarantees. The protocol has tricky edge cases.

Zero-knowledge authentication is not a silver bullet. It closes a specific, underappreciated gap — one that conventional hashing leaves wide open.

## Should You Use It?

For most web applications, adding SRP is non-trivial. It requires a compatible implementation on both client and server, and the ecosystem — while it exists — is not as mature as "npm install bcrypt".

Good candidates for adopting ZK auth:
- Applications with high-value credentials (financial, healthcare, identity providers)
- Internal tools where insider threat is a realistic concern
- Situations where regulatory or compliance requirements demand it

Available libraries include `thinbus-srp` for JavaScript, `srptools` for Python, and `srp` for Go, among others.

If you are building something new and the threat model fits, it is worth the integration cost. If you are maintaining a legacy system, the path of least resistance is ensuring your hashing is strong (Argon2id) and your server-side attack surface is minimized — but know that you are leaving this gap open.

## Conclusion

Hashing passwords with a unique salt is table stakes. It is necessary, it is not sufficient. The persistent server-side attacker — the insider, the persistent intruder — operates in a space that hashing cannot reach, because the plaintext still lands on the server on every login.

Zero-knowledge authentication closes that gap by ensuring the server never touches the password at all. The Schnorr-based scheme above demonstrates the core idea; production protocols like SRP and OPAQUE build on the same foundation while closing the engineering gaps.

In the next post, we will look at **OPAQUE** — the modern cryptographic successor to SRP — and **Passkeys / WebAuthn**, which bring this philosophy to mainstream consumer applications today.
