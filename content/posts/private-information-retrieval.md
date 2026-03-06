+++
date = '2026-03-06T15:17:16+02:00'
draft = false
title = 'Private Information Retrieval - Part 1'
tags = ['cryptography', 'privacy', 'pir', 'private information retrieval']
math = true
+++

Is privacy even real? We surf the web, watch videos, read articles, and search for specific information — all through web applications that process our requests and know exactly what we find interesting and what we love to read. A natural question arises: is it possible to expose a database on a server to a user in a way that lets the user read entries from the database without the server learning which entry was accessed? In 1998, Benny Chor, Oded Goldreich, Eyal Kushilevitz, and Madhu Sudan formalized this problem and named it Private Information Retrieval (PIR).

## Introduction
Consider a user who wants to read an entry from a database while keeping their query private from the server. A trivial solution is for the user to download the entire database instead of just the specific entry. Since the user downloads everything, the server cannot distinguish which entry they were interested in, and the user can then locally look up the relevant entry. Although this solution perfectly hides the query, it comes at a steep cost: the communication overhead equals the full size of the database. As it turns out, there are ways to achieve privacy with far better communication efficiency. As we will see, there are different PIR schemes — some require multiple servers, while others work with only a single server.

{{< definition title="Private Information Retrieval Scheme" >}}
We model the database as a string $\mathbf{x} = (x_1, x_2, \ldots, x_n) \in \{0,1\}^n$ of $n$ bits. A user wants to retrieve the bit $x_i$ for some secret index $i \in [n] = \{1, \ldots, n\}$.

A **$k$-server PIR scheme** involves $k$ servers, each holding an identical copy of $\mathbf{x}$, and a user who interacts with all $k$ servers. The protocol proceeds as follows:

1. The user, given index $i$, generates $k$ queries $(q_1, \ldots, q_k)$ and sends $q_j$ to server $j$.
2. Each server $j$ replies with an answer $a_j = \text{Answer}(\mathbf{x},q_j)$ computed from its query and the database.
3. The user reconstructs $x_i$ from the answers $(a_1, \ldots, a_k)$.

A PIR scheme must satisfy two properties:

**Correctness.** For every database $\mathbf{x}$ and every index $i$, the user always recovers $x_i$ correctly.

**1-Privacy.** For each server $j$, the query $q_j$ reveals nothing about which index the user is interested in.
{{< /definition >}}

The goal is to minimize the total **communication complexity** — the number of bits exchanged between the user and all servers — while achieving both correctness and privacy. Before building any schemes, it is worth taking a short detour to explain the distinction between information-theoretic privacy and computational privacy, since the two lead to very different constructions.

### Information-Theoretic Privacy

A PIR scheme is **information-theoretically (IT) private** if even a computationally unbounded adversary — one with unlimited time and memory, or even a quantum computer — cannot learn anything about the queried index from the query it receives. There is no assumption on the adversary's resources; the privacy guarantee holds unconditionally.

Formally, a PIR scheme is IT-private if for every server $j$ and every pair of indices $i, i' \in [n]$, the query distributions are identical:

$$q_j(i) \overset{d}{=} q_j(i')$$

This says that the query sent to server $j$ when the user wants index $i$ is distributed exactly the same as when the user wants index $i'$. Given only $q_j$, a server cannot do better than guessing the target index uniformly at random — there is simply no information to extract.

IT-privacy is a very strong guarantee, but as we will see, it comes with a cost: it requires multiple non-colluding servers.

### Computational Privacy

A PIR scheme is **computationally private** if no *efficient* adversary — one running in polynomial time — can distinguish between queries for different indices, except with negligible probability. Unlike IT-privacy, this guarantee is relative to a computational hardness assumption: we assume that some underlying mathematical problem (such as factoring large integers or solving certain lattice problems) is infeasible for polynomial-time algorithms.

Formally, instead of requiring identical distributions, we require that for every probabilistic polynomial-time (PPT) distinguisher $\mathcal{D}$ and every pair of indices $i, i' \in [n]$:

$$\left| \Pr[\mathcal{D}(q_j(i)) = 1] - \Pr[\mathcal{D}(q_j(i')) = 1] \right| \leq \varepsilon(n)$$

where $\varepsilon(n)$ is a negligible function — one that vanishes faster than any inverse polynomial.

This is a weaker guarantee than IT-privacy: a computationally unbounded adversary might, in principle, distinguish queries. But in practice, breaking the scheme is as hard as solving the underlying computational problem, which is believed to be infeasible. The payoff is significant: computational privacy makes **single-server PIR** possible, which is provably impossible under IT-privacy.


## A Single Server Is Not Enough (for IT-PIR)

Before constructing a clever scheme, it is worth asking whether we can do better than downloading the entire database with just one server — assuming we require **information-theoretic privacy**. The answer is no, and the argument is elegant.

{{< theorem >}}
Any 1-server IT-PIR scheme requires the server to respond with at least $n$ bits.
{{< /theorem >}}

{{< proof >}}
Suppose we have a 1-server IT-PIR scheme where the user sends a query $q$ to the server and the server responds with $a = \text{Answer}(\mathbf{x}, q)$. By IT-privacy, the distribution of $q$ must be identical for every index $i$ — no query carries any information about which index was targeted. Pick any query $q^*$ that is sent with positive probability; since $q^*$ is equally likely to have been generated for any index, the server's response to $q^*$ must allow the user to decode $x_i$ for *every* $i \in [n]$. Therefore, $a = \text{Answer}(\mathbf{x}, q^*)$ must encode all $n$ bits of the database. Since this holds for every possible query, the server's answer must always be at least $n$ bits long.
{{< /proof >}}

This impossibility result applies specifically to information-theoretic privacy. To achieve any meaningful communication savings with a single server, we must relax to computational privacy — which is exactly what computational PIR schemes do. For now, we focus on IT-PIR and see how using multiple **non-colluding** servers breaks the lower bound.

## Two-Server PIR: The Matrix Scheme

We now construct a concrete 2-server PIR scheme with communication complexity $O(\sqrt{n})$ — an exponential improvement over the trivial $O(n)$ bound.

### Setup

Arrange the $n$-bit database as a $\sqrt{n} \times \sqrt{n}$ matrix $X$ over $\mathbb{F}_2$ (assume $n$ is a perfect square for simplicity), where $X_{r,c} = x_{r,c}$. The user wants to retrieve $x_{r,c}$ — the bit at row $r$ and column $c$ — without revealing either index to either server.

### Protocol

**Query generation.** The user samples a uniformly random vector $\mathbf{v} \in \{0,1\}^{\sqrt{n}}$ and constructs two query vectors:

$$q_1 = \mathbf{v}, \qquad q_2 = \mathbf{v} \oplus \mathbf{e}_c$$

where $\mathbf{e}_c$ is the $c$-th standard basis vector (a $\sqrt{n}$-bit string that is $1$ at position $c$ and $0$ everywhere else), and $\oplus$ denotes addition over $\mathbb{F}_2$ (i.e., XOR). The user sends $q_1$ to server 1 and $q_2$ to server 2.

**Server response.** Each server $j$ multiplies the database matrix by its query vector over $\mathbb{F}_2$ and returns the resulting $\sqrt{n}$-bit vector:

$$\mathbf{a}_j = X \cdot q_j \in \{0,1\}^{\sqrt{n}}$$

The $r$-th entry of $\mathbf{a}_j$ is the inner product $\langle X_r,\, q_j \rangle \pmod{2}$, where $X_r$ is the $r$-th row of $X$.

**Reconstruction.** The user XORs the two response vectors and reads off the $r$-th entry:

$$x_{r,c} = (\mathbf{a}_1 \oplus \mathbf{a}_2)_r$$

### Correctness

Over $\mathbb{F}_2$, matrix-vector multiplication is linear, so:

$$\mathbf{a}_1 \oplus \mathbf{a}_2 = X\mathbf{v} \oplus X(\mathbf{v} \oplus \mathbf{e}_c) = X\mathbf{v} \oplus X\mathbf{v} \oplus X\mathbf{e}_c = X\mathbf{e}_c$$

Multiplying a matrix by $\mathbf{e}_c$ selects its $c$-th column, so $(X\mathbf{e}_c)_r = X_{r,c} = x_{r,c}$. ✓

### Privacy

- **Server 1** receives $q_1 = \mathbf{v}$, which is uniformly random in $\{0,1\}^{\sqrt{n}}$. Its distribution is independent of both $r$ and $c$. ✓
- **Server 2** receives $q_2 = \mathbf{v} \oplus \mathbf{e}_c$. Since $\mathbf{v}$ is uniform and XOR-ing with any fixed vector is a bijection on $\{0,1\}^{\sqrt{n}}$, $q_2$ is also uniformly distributed — regardless of $c$. ✓

Crucially, neither query encodes $r$ in any way, so the row index is never exposed to either server either.

### Communication Complexity

Each query $q_j$ is a $\sqrt{n}$-bit vector. Each response $\mathbf{a}_j$ is also a $\sqrt{n}$-bit vector. The total communication is:

$$2 \cdot \underbrace{\sqrt{n}}_{\text{query}} + 2 \cdot \underbrace{\sqrt{n}}_{\text{response}} = 4\sqrt{n} = O(\sqrt{n})$$

Compared to the trivial $n$-bit download, this is a dramatic reduction for large $n$.

## Where to Go From Here

The matrix scheme is simple and already achieves $O(\sqrt{n})$ communication with 2 servers. But it is far from the limit. Chor et al. showed that a more structured construction — arranging the database as a three-dimensional cube and using a carefully designed query strategy — achieves $O(n^{1/3})$ with 2 servers. More generally, with $k$ servers one can achieve $O(n^{1/(2k-1)})$ communication. Later work by Woodruff and Yekhanin pushed this further, and today there exist schemes with polylogarithmic communication, albeit under additional assumptions.

What about a single server? If we are willing to settle for **computational** rather than **information-theoretic** privacy — relying on cryptographic hardness assumptions rather than unconditional guarantees — then single-server PIR becomes possible. In Part 2, we will explore **computational PIR**, where constructions based on lattice problems like Learning With Errors (LWE) allow a single server to answer queries without learning the index, at the cost of heavier computation.

