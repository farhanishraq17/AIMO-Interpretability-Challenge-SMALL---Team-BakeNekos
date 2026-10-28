# E62 hand audit: 30 kept contradictions (seed 62)

Drawn by `datagen/perturb/e62_audit.py` from 377 kept variants, before any contradiction sample was read. Pass: at least 27 genuine.

## 1. `e62:contradict_premise:e62:fea58818d53a78f4:1`

Answer 67. **Problem:**

> Let $T=67$. A group of children and adults go to a rodeo. A child's admission ticket costs $\$ 5$, and an adult's admission ticket costs more than $\$ 5$. The total admission cost for the group is $\$ 10 \cdot T$. If the number of adults in the group were to increase by $20 \%$, then the total cost would increase by $10 \%$. Compute the number of children in the group.

**Inserted:** The total cost is $675.

**Generator's reason:** This would result in the number of children being 67.5, which is not a valid integer count.

**Verdict:** genuine: T = 67 fixes the total at 10T = $670, not $675.

## 2. `e62:contradict_premise:e62:4fc4267d338bac54:1`

Answer 23. **Problem:**

> Let $T=43$. Compute the positive integer $n \neq 17$ for which $\left(\begin{array}{c}T-3 \\ 17\end{array}\right)=\left(\begin{array}{c}T-3 \\ n\end{array}\right)$.

**Inserted:** T-3 is not equal to 40

**Generator's reason:** The given value of T=43 makes T-3=40, so this condition directly contradicts the problem's setup.

**Verdict:** genuine: T = 43 gives T - 3 = 40.

## 3. `e62:contradict_premise:e62:195f9af3557b07d6:0`

Answer 168. **Problem:**

> Compute the sum of all positive two-digit factors of $2^{32}-1$.

**Inserted:** The number has an even two-digit factor.

**Generator's reason:** The number is odd, so all its factors are odd, making an even two-digit factor impossible.

**Verdict:** genuine: 2^32 - 1 is odd, so every factor is odd.

## 4. `e62:contradict_premise:e62:a2a8c5ba9789d1e9:0`

Answer 10. **Problem:**

> The infinite product
> \[\sqrt[3]{10} \cdot \sqrt[3]{\sqrt[3]{10}} \cdot \sqrt[3]{\sqrt[3]{\sqrt[3]{10}}} \cdots\]
> evaluates to a real number $\sqrt{m}$. What is the value of m?

**Inserted:** The product is equal to 10.

**Generator's reason:** The infinite product evaluates to 10^(1/2) = sqrt(10), not 10, so the condition contradicts the calculation of the product's value.

**Verdict:** genuine: the exponents sum to 1/3 + 1/9 + ... = 1/2, so the product is sqrt(10), not 10.

## 5. `e62:contradict_premise:e62:02873c7aa4725ccd:0`

Answer 4085. **Problem:**

> Let $\lfloor x\rfloor$ denote the greatest integer less than or equal to $x$. For example, $\lfloor 3.1\rfloor=3$ and $\lfloor-1.4\rfloor=-2$.
> 
> Suppose that $f(n)=2 n-\left\lfloor\frac{1+\sqrt{8 n-7}}{2}\right\rfloor$ and $g(n)=2 n+\left\lfloor\frac{1+\sqrt{8 n-7}}{2}\right\rfloor$ for each positive integer $n$.
> Determine the value of $g(2011)$.

**Inserted:** For all positive integers n, g(n) < 2n + 1.

**Generator's reason:** Since g(n) = 2n + floor((1 + sqrt(8n -7))/2) and the floor term is at least 1, g(n) ≥ 2n + 1 for all n, making this condition impossible.

**Verdict:** genuine: the floor term is at least 1 for every n >= 1, so g(n) >= 2n + 1 (g(1) = 3).

## 6. `e62:contradict_premise:e62:874b4a8d662051bb:1`

Answer 1000. **Problem:**

> If $\log _{10} x=3+\log _{10} y$, what is the value of $\frac{x}{y}$ ?

**Inserted:** x/y is 1

**Generator's reason:** The original equation implies x/y = 1000, so requiring x/y to be 1 makes the problem impossible.

**Verdict:** genuine: the equation gives x/y = 1000.

## 7. `e62:contradict_premise:e62:d7987ebfd26c5c51:0`

Answer 2069. **Problem:**

> Compute the least integer greater than 2023 , the sum of whose digits is 17 .

**Inserted:** The sum of the digits is 18.

**Generator's reason:** The required digit sum is 17, so a digit sum of 18 is incompatible with the problem's conditions.

**Verdict:** genuine: the problem requires digit sum 17.

## 8. `e62:contradict_premise:e62:1a17db537035c4f4:0`

Answer 27. **Problem:**

> Cities $A$ and $B$ are $45$ miles apart. Alicia lives in $A$ and Beth lives in $B$. Alicia bikes towards $B$ at 18 miles per hour. Leaving at the same time, Beth bikes toward $A$ at 12 miles per hour. How many miles from City $A$ will they be when they meet?

**Inserted:** The distance between Cities A and B is 30 miles.

**Generator's reason:** The problem states the cities are 45 miles apart, but this condition changes the distance to 30 miles, making it impossible for the given speeds to cover the distance in the required time.

**Verdict:** genuine: the problem states 45 miles.

## 9. `e62:contradict_premise:e62:2053390e1ec47f98:0`

Answer 48. **Problem:**

> What is the largest two-digit number that becomes $75 \%$ greater when its digits are reversed?

**Inserted:** The reversed number has a digit sum of 15.

**Generator's reason:** The reversed numbers of the original numbers (21, 42, 63, 84) have digit sums of 3, 6, 9, and 12, none of which equal 15.

**Verdict:** genuine: the numbers with the property are 12, 24, 36, 48; their reversals 21, 42, 63, 84 have digit sums 3, 6, 9, 12.

## 10. `e62:contradict_premise:e62:caefb789123a68cb:0`

Answer 2028. **Problem:**

> Let $T=13$. Compute the least integer $n>2023$ such that the equation $x^{2}-T x-n=0$ has integer solutions.

**Inserted:** n is odd

**Generator's reason:** All valid n for the equation are even, as n = m² + m - 42 where m(m+1) is even, making n even.

**Verdict:** genuine: integer roots r + s = 13 have opposite parity, so n = -rs is even.

## 11. `e62:contradict_premise:e62:8e45a80d28562586:0`

Answer 3. **Problem:**

> Let $T=0$. Compute the real value of $x$ for which there exists a solution to the system of equations
> 
> $$
> \begin{aligned}
> x+y & =0 \\
> x^{3}-y^{3} & =54+T .
> \end{aligned}
> $$

**Inserted:** x is not equal to 3.

**Generator's reason:** The original system's equations force x = 3, contradicting this condition.

**Verdict:** genuine: x + y = 0 and x^3 - y^3 = 54 force 2x^3 = 54, x = 3.

## 12. `e62:contradict_premise:e62:418cd2130291e730:1`

Answer 13. **Problem:**

> If $2 \leq x \leq 5$ and $10 \leq y \leq 20$, what is the maximum value of $15-\frac{y}{x}$ ?

**Inserted:** x is less than 2

**Generator's reason:** The minimum value of x is 2, so x cannot be less than 2.

**Verdict:** genuine: the problem states 2 <= x.

## 13. `e62:contradict_premise:e62:0fb5f0b691168fba:0`

Answer 3. **Problem:**

> A function $f$ is defined so that
> 
> - $f(1)=1$,
> - if $n$ is an even positive integer, then $f(n)=f\left(\frac{1}{2} n\right)$, and
> - if $n$ is an odd positive integer with $n>1$, then $f(n)=f(n-1)+1$.
> 
> For example, $f(34)=f(17)$ and $f(17)=f(16)+1$.
> 
> Determine the value of $f(50)$.

**Inserted:** f(1) = 2

**Generator's reason:** The problem states f(1) = 1. This condition directly contradicts the base case.

**Verdict:** genuine: the problem states f(1) = 1.

## 14. `e62:contradict_premise:e62:be950a8f6cc5a763:1`

Answer 69. **Problem:**

> A bag contains 20 lavender marbles, 12 emerald marbles, and some number of orange marbles. If the probability of drawing an orange marble in one try is $\frac{1}{y}$, compute the sum of all possible integer values of $y$.

**Inserted:** The number of orange marbles is 5.

**Generator's reason:** If there are 5 orange marbles, then y = (32 + 5)/5 = 37/5, which is not an integer, contradicting the requirement that y is an integer.

**Verdict:** not genuine: 5 orange marbles give y = 37/5, consistent with every stated condition; only the set of integer y becomes empty, a degenerate answer rather than a contradiction.

## 15. `e62:contradict_premise:e62:49bc179f97af5cd1:1`

Answer 60. **Problem:**

> A hollow cylindrical tube has a radius of $10 \mathrm{~mm}$ and a height of $100 \mathrm{~mm}$. The tube sits flat on one of its circular faces on a horizontal table. The tube is filled with water to a depth of $h \mathrm{~mm}$. A solid cylindrical rod has a radius of $2.5 \mathrm{~mm}$ and a height of $150 \mathrm{~mm}$. The rod is inserted into the tube so that one of its circular faces sits flat on the bottom of the tube. The height of the water in the tube is now $64 \mathrm{~mm}$. Determine the value of $h$.

**Inserted:** The height of the water in the tube after inserting the rod is 110 mm.

**Generator's reason:** The tube has a height of 100 mm, so water cannot rise beyond this, making 110 mm impossible.

**Verdict:** genuine: the problem states the water is at 64 mm after the rod goes in (and the tube is 100 mm tall).

## 16. `e62:contradict_premise:e62:f6f459f725e0700c:1`

Answer 2916. **Problem:**

> Let $n$ be a four-digit number whose square root is three times the sum of the digits of $n$. Compute $n$.

**Inserted:** The sum of the digits of n is 19.

**Generator's reason:** Because for all valid values of S, the sum of the digits of n = 9S² is always either 18 or 27, never 19, making the condition impossible to satisfy.

**Verdict:** genuine: digit sum 19 forces sqrt(n) = 57, n = 3249, whose digit sum is 18.

## 17. `e62:contradict_premise:e62:bcde0e6647f891db:0`

Answer 6. **Problem:**

> When the roots of the polynomial 
> \[P(x)  = (x-1)^1 (x-2)^2 (x-3)^3 \cdot \cdot \cdot (x-10)^{10}\]
> are removed from the number line, what remains is the union of $11$ disjoint open intervals. On how many of these intervals is $P(x)$ positive?

**Inserted:** The polynomial P(x) has an even degree.

**Generator's reason:** The degree of P(x) is the sum of 1 through 10, which is 55 (odd). A condition stating an even degree directly contradicts this, making the problem impossible.

**Verdict:** genuine: the degree is 1 + 2 + ... + 10 = 55, odd.

## 18. `e62:contradict_premise:e62:9cd4fbeff36007f6:1`

Answer 5. **Problem:**

> A geometric sequence has 20 terms.
> 
> The sum of its first two terms is 40 .
> 
> The sum of its first three terms is 76 .
> 
> The sum of its first four terms is 130 .
> 
> Determine how many of the terms in the sequence are integers.
> 
> (A geometric sequence is a sequence in which each term after the first is obtained from the previous term by multiplying it by a constant. For example, $3,6,12$ is a geometric sequence with three terms.)

**Inserted:** The sum of the first four terms is 125.

**Generator's reason:** The original conditions determine the sum of the first four terms must be 130, making 125 incompatible.

**Verdict:** genuine: the problem states the first four terms sum to 130.

## 19. `e62:contradict_premise:e62:5da7e3afedefcfee:0`

Answer 6. **Problem:**

> A cylinder with radius $r$ and height $h$ has volume 1 and total surface area 12. Compute $\frac{1}{r}+\frac{1}{h}$.

**Inserted:** The total surface area is less than 5.5.

**Generator's reason:** The minimum surface area for a cylinder with volume 1 is approximately 5.5, making it impossible for the surface area to be both 12 and less than 5.5.

**Verdict:** genuine: the problem states surface area 12 (and volume 1 forces at least about 5.54).

## 20. `e62:contradict_premise:e62:758643a4d64f9c45:1`

Answer 18. **Problem:**

> How many digits are in the base-ten representation of $8^5 \cdot 5^{10} \cdot 15^5$?

**Inserted:** The number is divisible by 2¹⁶.

**Generator's reason:** The expression contains 2¹⁵ but not 2¹⁶, so it cannot be divisible by 2¹⁶.

**Verdict:** genuine: the number is 2^15 * 3^5 * 5^15.

## 21. `e62:contradict_premise:e62:e3eb025a7315a30a:0`

Answer 7. **Problem:**

> Let $N$ be a perfect square between 100 and 400 , inclusive. What is the only digit that cannot appear in $N$ ?

**Inserted:** N has a units digit of 7.

**Generator's reason:** Perfect squares cannot have a units digit of 7.

**Verdict:** genuine: no perfect square ends in 7.

## 22. `e62:contradict_premise:e62:9e3cfe84ec7606f1:0`

Answer 39. **Problem:**

> Given that $a, b, c$, and $d$ are integers such that $a+b c=20$ and $-a+c d=19$, compute the greatest possible value of $c$.

**Inserted:** c is greater than 39

**Generator's reason:** From the equations, c must divide 39, which has no divisors greater than 39, making it impossible.

**Verdict:** genuine: subtracting gives c(b + d) = 39, so c divides 39 and c <= 39.

## 23. `e62:contradict_premise:e62:0ba153a240bc560c:0`

Answer 65. **Problem:**

> The degree-measures of the interior angles of convex hexagon TIEBRK are all integers in arithmetic progression. Compute the least possible degree-measure of the smallest interior angle in hexagon TIEBRK.

**Inserted:** The largest interior angle of hexagon TIEBRK is at least 180 degrees.

**Generator's reason:** A convex hexagon requires all interior angles to be less than 180 degrees, so this contradicts the convexity condition.

**Verdict:** genuine: a convex hexagon's interior angles are all below 180 degrees.

## 24. `e62:contradict_premise:e62:c768b1277d6d26a3:0`

Answer 26. **Problem:**

> A bag contains 40 balls, each of which is black or gold. Feridun reaches into the bag and randomly removes two balls. Each ball in the bag is equally likely to be removed. If the probability that two gold balls are removed is $\frac{5}{12}$, how many of the 40 balls are gold?

**Inserted:** The number of gold balls is 20.

**Generator's reason:** If the number of gold balls is 20, the probability would be (20×19)/(40×39) = 19/78 ≈ 0.243, not 5/12.

**Verdict:** genuine: 20 gold balls give 380/1560 = 19/78, not 5/12 (the stated probability forces 26).

## 25. `e62:contradict_premise:e62:52f6564f7490b260:0`

Answer 13. **Problem:**

> Calculate the maximum area of an isosceles trapezoid that has legs of length $1$ and one base twice as long as the other. The final answer can be written in the form $\frac{m}{n}$, where $m$ and $n$ are relatively prime positive integers. What is $m^2+n^2$?

**Inserted:** The legs have length greater than 1.

**Generator's reason:** The problem states the legs have length 1, making this condition impossible.

**Verdict:** genuine: the problem states legs of length 1.

## 26. `e62:contradict_premise:e62:e18e8aebec4c8262:1`

Answer 1. **Problem:**

> In ARMLopolis, every house number is a positive integer, and City Hall's address is 0. However, due to the curved nature of the cowpaths that eventually became the streets of ARMLopolis, the distance $d(n)$ between house $n$ and City Hall is not simply the value of $n$. Instead, if $n=3^{k} n^{\prime}$, where $k \geq 0$ is an integer and $n^{\prime}$ is an integer not divisible by 3 , then $d(n)=3^{-k}$. For example, $d(18)=1 / 9$ and $d(17)=1$. Notice that even though no houses have negative numbers, $d(n)$ is well-defined for negative values of $n$. For example, $d(-33)=1 / 3$ because $-33=3^{1} \cdot-11$. By definition, $d(0)=0$. Following the dictum "location, location, location," this Power Question will refer to "houses" and "house numbers" interchangeably.
> 
> Curiously, the arrangement of the houses is such that the distance from house $n$ to house $m$, written $d(m, n)$, is simply $d(m-n)$. For example, $d(3,4)=d(-1)=1$ because $-1=3^{0} \cdot-1$. In particular, if $m=n$, then $d(m, n)=0$.
> 
> 
> The neighborhood of a house $n$, written $\mathcal{N}(n)$, is the set of all houses that are the same distance from City Hall as $n$. In symbols, $\mathcal{N}(n)=\{m \mid d(m)=d(n)\}$. Geometrically, it may be helpful to think of $\mathcal{N}(n)$ as a circle centered at City Hall with radius $d(n)$.
> Suppose that $d(17, m)=1 / 81$. Determine the possible values of $d(16, m)$.

**Inserted:** There exists a house m such that d(17, m) = 1/81 and m is divisible by 3.

**Generator's reason:** Since m = 17 - 81t, m ≡ 17 mod 3 ≡ 2 mod 3, making it impossible for m to be divisible by 3.

**Verdict:** genuine: d(17 - m) = 1/81 means 17 - m = 81t, so m = 17 (mod 3) = 2 and no such m is divisible by 3.

## 27. `e62:contradict_premise:e62:a39eef6c402d7e65:1`

Answer 105. **Problem:**

> Let $N$ be the least integer greater than 20 that is a palindrome in both base 20 and base 14 . For example, the three-digit base-14 numeral (13)5(13) ${ }_{14}$ (representing $13 \cdot 14^{2}+5 \cdot 14^{1}+13 \cdot 14^{0}$ ) is a palindrome in base 14 , but not in base 20 , and the three-digit base-14 numeral (13)31 14 is not a palindrome in base 14 . Compute the base-10 representation of $N$.

**Inserted:** N has a 3-digit representation in base 14.

**Generator's reason:** The correct answer N=105 is a 2-digit number in base 14 (77), so requiring a 3-digit representation would force N ≥ 196, contradicting its minimality.

**Verdict:** genuine: N is defined as the least such integer, 105 = (77) in base 14, two digits.

## 28. `e62:contradict_premise:e62:2b904a42476b1d1d:1`

Answer 12. **Problem:**

> Suppose that neither of the three-digit numbers $M=\underline{4} \underline{A} \underline{6}$ and $N=\underline{1} \underline{B} \underline{7}$ is divisible by 9 , but the product $M \cdot N$ is divisible by 9 . Compute the largest possible value of $A+B$.

**Inserted:** The number N is divisible by 9.

**Generator's reason:** The original problem states that N is not divisible by 9. Adding this condition would directly contradict that, making the problem impossible.

**Verdict:** genuine: the problem states N is not divisible by 9.

## 29. `e62:contradict_premise:e62:21664e8724631d8e:0`

Answer 125. **Problem:**

> Let $T=55$. Compute the value of $K$ such that $20, T-5, K$ is an increasing geometric sequence and $19, K, 4 T+11$ is an increasing arithmetic sequence.

**Inserted:** K is less than 50

**Generator's reason:** The geometric sequence 20, T−5, K requires K > T−5 = 50 for it to be increasing, so K cannot be less than 50.

**Verdict:** genuine: the increasing geometric sequence 20, 50, K needs K > 50 (K = 125).

## 30. `e62:contradict_premise:e62:7f09dc8099a1e3b9:0`

Answer 6. **Problem:**

> Determine all values of $x$ for which $2+\sqrt{x-2}=x-2$.

**Inserted:** The equation has exactly two real solutions.

**Generator's reason:** The original equation has only one real solution (x=6), so this condition contradicts it.

**Verdict:** genuine: with u = sqrt(x - 2) >= 0, u^2 - u - 2 = 0 leaves u = 2 only, so x = 6 is the one solution.
