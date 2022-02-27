(set-logic CHC_LIA)

(synth-fun main@NewDefault.outer.i ((x_0 Int) (x_1 Int) (x_2 Bool) (x_3 Int) (x_4 Int)) Bool)

(synth-fun main@entry () Bool)

(synth-fun main@NewDefault.i.us ((x_0 Int) (x_1 Int) (x_2 Bool) (x_3 Int)) Bool)

(synth-fun main@verifier.error.split () Bool)

(synth-fun main@NewDefault.i.us5 ((x_0 Int) (x_1 Int) (x_2 Bool) (x_3 Int) (x_4 Int)) Bool)

(synth-fun main@_bb14 ((x_0 Int) (x_1 Int) (x_2 Int) (x_3 Int) (x_4 Bool) (x_5 Int) (x_6 Int) (x_7 Int)) Bool)

(constraint (forall ((CHC_COMP_UNUSED Bool)) (=> (and true) main@entry)))
(constraint (forall ((A Int) (B Bool) (C Int) (D Int) (E Int) (F Int) (G Bool) (H Bool) (I Bool) (J Int) (K Int) (L Int) (M Bool) (N Int) (O Int)) (=> (and main@entry (and (= J 0) (= C D) (= D (+ 296 O)) (= E 0) (= F 0) (or (not H) (not I) (= M G)) (or (not H) (not I) (= K E)) (or (not H) (not I) (= L F)) (or (not H) (not I) (= N J)) (or (<= O 0) (not (<= D 0))) (or (not H) (and H I)) (= B true) (= H true) (not G) (= B (= A 0)))) (main@NewDefault.outer.i K L M N O))))
(constraint (forall ((A Bool) (B Bool) (C Bool) (D Int) (E Bool) (F Bool) (G Bool) (H Bool) (I Int) (J Int) (K Bool) (L Int)) (=> (and (main@NewDefault.outer.i I J K D L) (and (= F (= D 1)) (or (not E) F (not G)) (or (not E) (not B) (not C)) (or (not G) (and E G)) (or (not B) (and B A)) (or (not H) (and H G)) (or (not E) (and E B)) (= H true) (= C (not (<= 1 D))))) (main@NewDefault.i.us I J K L))))
(constraint (forall ((A Bool) (B Bool) (C Bool) (D Bool) (E Bool) (F Bool) (G Bool) (H Int) (I Int) (J Bool) (K Int) (L Int)) (=> (and (main@NewDefault.outer.i H I J K L) (and (= E (= K 0)) (or (not D) (not B) C) (or (not D) E (not F)) (or (not G) (and F G)) (or (not F) (and D F)) (or (not B) (and B A)) (or (not D) (and D B)) (= G true) (= C (not (<= 1 K))))) (main@NewDefault.i.us5 H I J K L))))
(constraint (forall ((A Int) (B Bool) (C Bool) (D Int) (E Int) (F Bool) (G Int) (H Int) (I Bool) (J Bool) (K Int) (L Int) (M Int) (N Bool) (O Int) (P Int) (Q Int) (R Bool) (S Bool) (T Bool) (U Int) (V Int) (W Int) (X Bool) (Y Int) (Z Int)) (=> (and (main@NewDefault.i.us D E F Z) (and (= C (= A 1)) (= R N) (= G D) (= U O) (= H E) (= K 0) (= P L) (= Q M) (or (not J) (not T) (= N I)) (or (not J) (not T) (= O K)) (or (not J) (not T) (= L G)) (or (not J) (not T) (= M H)) (or (not J) C (not B)) (or (not S) (not T) (= X R)) (or (not S) (not T) (= V P)) (or (not S) (not T) (= W Q)) (or (not S) (not T) (= Y U)) (or (not T) (and J T)) (or (not J) (and J B)) (or (not S) (and S T)) (= S true) (= I F))) (main@NewDefault.outer.i V W X Y Z))))
(constraint (forall ((A Int) (B Bool) (C Bool) (D Bool) (E Int) (F Int) (G Bool) (H Int)) (=> (and (main@NewDefault.i.us E F G H) (and (or (not D) (not B) (not C)) (or (not B) (and B C)) (= B true) (= D (= A 1)))) (main@NewDefault.i.us E F G H))))
(constraint (forall ((A Int) (B Bool) (C Int) (D Bool) (E Bool) (F Int) (G Int) (H Int) (I Int) (J Bool) (K Int) (L Int) (M Int) (N Int) (O Bool) (P Bool) (Q Int) (R Int) (S Int) (T Bool) (U Bool) (V Bool) (W Int) (X Int) (Y Int) (Z Int) (A1 Int) (B1 Int) (C1 Int) (D1 Bool) (E1 Bool) (F1 Int) (G1 Int) (H1 Bool) (I1 Int) (J1 Int) (K1 Bool) (L1 Bool) (M1 Int) (N1 Int) (O1 Int) (P1 Bool) (Q1 Bool) (R1 Int) (S1 Int) (T1 Int) (U1 Bool) (V1 Bool) (W1 Bool) (X1 Int) (Y1 Int) (Z1 Int) (A2 Bool) (B2 Bool) (C2 Bool) (D2 Int) (E2 Int) (F2 Bool) (G2 Int) (H2 Int) (I2 Int) (J2 Bool) (K2 Int) (L2 Int) (M2 Int) (N2 Bool) (O2 Bool) (P2 Int) (Q2 Int) (R2 Int) (S2 Bool) (T2 Int) (U2 Int) (V2 Int) (W2 Bool) (X2 Bool) (Y2 Bool) (Z2 Int) (A3 Int) (B3 Int) (C3 Bool) (D3 Int) (E3 Int)) (=> (and (main@NewDefault.i.us5 F1 G1 H1 G2 E3) (let ((a!1 (ite (>= H 0) (or (not (<= I H)) (not (>= I 0))) (and (not (<= I H)) (not (<= 0 I))))) (a!2 (ite (>= R 0) (or (not (<= S R)) (not (>= S 0))) (and (not (<= S R)) (not (<= 0 S)))))) (and (= J a!1) (= N2 J2) (= D1 (= Y 0)) (= P1 H1) (= E (= C 0)) (= V (not T)) (= P (not J)) (= K1 H1) (= B (not (= A 0))) (= U1 H1) (= F2 (= E2 0)) (= E1 (= F 0)) (= W2 S2) (= A2 H1) (= S1 F1) (= X1 (- 12)) (= R 2012) (= D2 (- 22)) (= I1 F1) (= K2 (+ 1 G2)) (= T1 G1) (= W X) (= C1 N) (= K L) (= L2 H2) (= Z2 T2) (= R1 C1) (= M2 I2) (= L (+ 168 Y)) (= Y1 F1) (= P2 K2) (= A1 (+ 2012 (* (- 1) Z))) (= O1 G1) (= M1 B1) (= J1 G1) (= N1 F1) (= U2 Q2) (= G L) (= Z1 G1) (= B1 A1) (= V2 R2) (= H 2012) (= N (+ 2012 (* (- 1) M))) (= X (+ 176 Y)) (= Q X) (or (not O2) (and W1 V1) (and O2 L1) (and O2 Q1) (and C2 B2)) (or (not U) (not (<= X 0)) (<= Y 0)) (or (not D) E (not B2)) (or (not L1) (not V) (not U)) (or (not C2) (not B2) (= J2 A2)) (or (not C2) (not B2) (= H2 Y1)) (or (not C2) (not B2) (= E2 D2)) (or (not C2) (not B2) (= I2 Z1)) (or (not O2) (not Q1) (= J2 P1)) (or (not O2) (not Q1) (= H2 N1)) (or (not O2) (not Q1) (= E2 R1)) (or (not O2) (not Q1) (= I2 O1)) (or (not O2) (not Y2) (= S2 N2)) (or (not O2) (not Y2) (= T2 P2)) (or (not O2) (not Y2) (= Q2 L2)) (or (not O2) (not Y2) (= R2 M2)) (or (not O2) (not L1) (= J2 K1)) (or (not O2) (not L1) (= H2 I1)) (or (not O2) (not L1) (= E2 M1)) (or (not O2) (not L1) (= I2 J1)) (or (not X2) (not Y2) (= C3 W2)) (or (not X2) (not Y2) (= A3 U2)) (or (not X2) (not Y2) (= B3 V2)) (or (not X2) (not Y2) (= D3 Z2)) (or (not W1) (not V1) (= J2 U1)) (or (not W1) (not V1) (= H2 S1)) (or (not W1) (not V1) (= E2 X1)) (or (not W1) (not V1) (= I2 T1)) (or (not W1) D1 (not V1)) (or (not O) (<= Y 0) (not (<= L 0))) (or (not O) (not D1) (not V1)) (or (not O) P (not U)) (or (not O) (not P) (not Q1)) (or (not E1) (not B2) (not V1)) (or E1 (not C2) (not B2)) (or (not U) (not (<= Y 0))) (or (not U) (and O U)) (or (not V1) (and B2 V1)) (or (not Q1) (not (<= Y 0))) (or (not Q1) (and O Q1)) (or (not Y2) (and O2 Y2)) (or (not B2) (not (<= E3 0))) (or (not B2) (and D B2)) (or (not L1) (not (<= Y 0))) (or (not L1) (and L1 U)) (or (not C2) B2) (or (not X2) (and X2 Y2)) (or (not W1) V1) (or F2 (not O2)) (or (not O) (not (<= Y 0))) (or (not O) (and O V1)) (= B true) (= X2 true) (= T a!2)))) (main@NewDefault.outer.i A3 B3 C3 D3 E3))))
(constraint (forall ((A Int) (B Bool) (C Int) (D Bool) (E Bool) (F Bool) (G Int) (H Int) (I Bool) (J Int) (K Int)) (=> (and (main@NewDefault.i.us5 G H I J K) (and (= F (= C 0)) (or (not D) (not E) (not F)) (or (not D) (and D E)) (= B true) (= D true) (= B (not (= A 0))))) (main@NewDefault.i.us5 G H I J K))))
(constraint (forall ((A Int) (B Bool) (C Int) (D Bool) (E Bool) (F Bool) (G Bool) (H Bool) (I Bool) (J Int) (K Int) (L Int) (M Int) (N Bool) (O Bool) (P Bool) (Q Int) (R Int) (S Int) (T Int) (U Bool) (V Bool) (W Bool) (X Int) (Y Int) (Z Int) (A1 Int) (B1 Bool) (C1 Int) (D1 Int) (E1 Bool) (F1 Bool) (G1 Bool) (H1 Int) (I1 Int) (J1 Int) (K1 Int) (L1 Int) (M1 Bool) (N1 Int) (O1 Int) (P1 Int)) (=> (and (main@NewDefault.i.us5 Z A1 B1 I1 P1) (let ((a!1 (ite (>= S 0) (or (not (<= T S)) (not (>= T 0))) (and (not (<= T S)) (not (<= 0 T))))) (a!2 (ite (>= L 0) (or (not (<= M L)) (not (>= M 0))) (and (not (<= M L)) (not (<= 0 M)))))) (and (= U a!1) (= E (= C 0)) (= G (= O1 0)) (= E1 B1) (= B (not (= A 0))) (= W (not U)) (= N a!2) (= P (not N)) (= Q R) (= Y (+ 184 X)) (= H1 0) (= D1 A1) (= J K) (= L 2012) (= S 2012) (= C1 Z) (= K (+ 168 X)) (= R (+ 176 X)) (= N1 Y) (or (not G1) (<= X 0) (not (<= Y 0))) (or (not V) (not (<= R 0)) (<= X 0)) (or W (not V) (not G1)) (or (not O) (not (<= K 0)) (<= X 0)) (or (not F1) (not G1) (= M1 E1)) (or (not F1) (not G1) (= J1 H1)) (or (not F1) (not G1) (= K1 C1)) (or (not F1) (not G1) (= L1 D1)) (or (not F) (not D) E) (or (not H) (not O) (not I)) (or (not H) (not F) (not G)) (or P (not O) (not V)) (or (not G1) (and V G1)) (or (not V) (not (<= X 0))) (or (not V) (and O V)) (or (not O) (not (<= X 0))) (or (not O) (and H O)) (or (not F1) (and F1 G1)) (or (not F) (not (<= P1 0))) (or (not F) (and F D)) (or (not H) (and H F)) (= B true) (= F1 true) (= I (= X 0))))) (main@_bb14 I1 J1 K1 L1 M1 N1 O1 P1))))
(constraint (forall ((A Int) (B Int) (C Int) (D Int) (E Int) (F Int) (G Int) (H Int) (I Int) (J Int) (K Int) (L Int) (M Bool) (N Int) (O Bool) (P Int) (Q Bool) (R Bool) (S Bool) (T Int) (U Bool) (V Bool) (W Bool) (X Int) (Y Int) (Z Int) (A1 Int) (B1 Int) (C1 Int) (D1 Int) (E1 Int) (F1 Bool) (G1 Int) (H1 Bool) (I1 Int) (J1 Bool) (K1 Int) (L1 Int) (M1 Bool) (N1 Bool) (O1 Bool) (P1 Int) (Q1 Int) (R1 Bool) (S1 Int) (T1 Int) (U1 Bool) (V1 Bool) (W1 Bool) (X1 Int) (Y1 Int) (Z1 Bool) (A2 Bool) (B2 Bool) (C2 Int) (D2 Int) (E2 Bool) (F2 Bool) (G2 Bool) (H2 Int) (I2 Int) (J2 Bool) (K2 Bool) (L2 Bool) (M2 Int) (N2 Int) (O2 Bool) (P2 Bool) (Q2 Bool) (R2 Int) (S2 Int) (T2 Bool) (U2 Bool) (V2 Int) (W2 Int) (X2 Bool) (Y2 Int) (Z2 Int) (A3 Bool) (B3 Bool) (C3 Int) (D3 Int) (E3 Bool) (F3 Int) (G3 Int) (H3 Int) (I3 Bool) (J3 Int) (K3 Int) (L3 Int) (M3 Bool) (N3 Bool) (O3 Int) (P3 Int) (Q3 Int) (R3 Bool) (S3 Int) (T3 Int) (U3 Int) (V3 Bool) (W3 Bool) (X3 Bool) (Y3 Int) (Z3 Int) (A4 Int) (B4 Bool) (C4 Int) (D4 Int)) (=> (and (main@_bb14 F3 R2 P1 Q1 R1 E1 B D4) (let ((a!1 (= U2 (and (not (<= 8 S2)) (>= S2 0))))) (and (= M (not (<= K1 Q1))) (= M3 I3) a!1 (= A3 X2) (= O1 (= F 0)) (= W R1) (= L2 R1) (= O (not R1)) (= F1 (not (<= I1 L1))) (= J1 (= X 0)) (= E3 (= D3 0)) (= Q2 R1) (= G2 M1) (= N1 (= E 0)) (= V3 R3) (= H1 (not M1)) (= B1 (+ 64 E1 (* 136 C1))) (= C3 0) (= H2 P1) (= I (+ 112 E1 (* 136 C1))) (= J3 (+ 1 F3)) (= K (+ 8 E1 (* 136 C1))) (= S2 (+ 1 R2)) (= S1 I1) (= T1 G1) (= K3 G3) (= Y3 S3) (= D (+ 12 E)) (= N (+ 1 Q1)) (= Y (+ 104 E1 (* 136 C1))) (= L3 H3) (= H (+ 48 E1 (* 136 C1))) (= K1 (+ 1 P1)) (= O3 J3) (= I1 (+ 1 K1)) (= Z (+ 124 E1 (* 136 C1))) (= N2 Q1) (= A1 (+ 56 E1 (* 136 C1))) (= Z2 W2) (= D1 (+ 88 E1 (* 136 C1))) (= C1 G) (= C2 K1) (= G R2) (= T Q1) (= A (+ B (* 8 R2))) (= Y1 L1) (= J (+ E1 (* 136 C1))) (= I2 Q1) (= M2 P1) (= T3 P3) (= Y2 V2) (= C E) (= X1 I1) (= L (+ 32 E1 (* 136 C1))) (= P N) (= U3 Q3) (= G1 (+ 1 L1)) (= D2 L1) (not (<= B 0)) (or (not T2) (and V1 U1) (and F2 E2) (and K2 J2) (and O2 P2) (and Z1 A2)) (or (not Z1) (<= E1 0) (not (<= B1 0))) (or (not Z1) (not (<= Y 0)) (<= E1 0)) (or (not Z1) (not (<= Z 0)) (<= E1 0)) (or (not Z1) (not (<= A1 0)) (<= E1 0)) (or (not Z1) (not (<= D1 0)) (<= E1 0)) (or (not Z1) (not A2) (= X2 B2)) (or (not Z1) (not A2) (= W2 Y1)) (or (not Z1) (not A2) (= V2 X1)) (or (not O2) (<= E 0) (not (<= D 0))) (or (not O2) (not P2) (= X2 Q2)) (or (not O2) (not P2) (= W2 N2)) (or (not O2) (not P2) (= V2 M2)) (or O1 (not O2) (not P2)) (or (not E2) (and V U) (and Q R)) (or (not K2) (not J2) (= X2 L2)) (or (not K2) (not J2) (= W2 I2)) (or (not K2) (not J2) (= V2 H2)) (or (not T2) (not B3) (not U2)) (or (not Q) (not R) (= M1 S)) (or (not Q) (not R) (= L1 P)) (or (not Q) (not R) M) (or (not U) (<= E1 0) (not (<= I 0))) (or (not U) (<= E1 0) (not (<= K 0))) (or (not U) (not (<= H 0)) (<= E1 0)) (or (not U) (not (<= J 0)) (<= E1 0)) (or (not U) (not (<= L 0)) (<= E1 0)) (or (not U) (not O1) (not O2)) (or (not V) (not U) (= M1 W)) (or (not V) (not U) (= L1 T)) (or (not N3) (not X3) (= R3 M3)) (or (not N3) (not X3) (= S3 O3)) (or (not N3) (not X3) (= P3 K3)) (or (not N3) (not X3) (= Q3 L3)) (or (not N3) (not B3) (= I3 A3)) (or (not N3) (not B3) (= G3 Y2)) (or (not N3) (not B3) (= D3 C3)) (or (not N3) (not B3) (= H3 Z2)) (or (not F2) (not E2) (= X2 G2)) (or (not F2) (not E2) (= W2 D2)) (or (not F2) (not E2) (= V2 C2)) (or O (not U) (not Q)) (or (not O) (not V) (not U)) (or (not W3) (not X3) (= B4 V3)) (or (not W3) (not X3) (= Z3 T3)) (or (not W3) (not X3) (= A4 U3)) (or (not W3) (not X3) (= C4 Y3)) (or (not J1) (not E2) (not Z1)) (or J1 (not F2) (not E2)) (or (not N1) (not J2) (not O2)) (or N1 (not K2) (not J2)) (or (not V1) (not U1) (= X2 W1)) (or (not V1) (not U1) (= W2 T1)) (or (not V1) (not U1) (= V2 S1)) (or (not V1) F1 (not U1)) (or (not H1) (not Z1) (not A2)) (or H1 (not U1) (not Z1)) (or (not (<= A 0)) (<= B 0)) (or (not Z1) (not (<= E1 0))) (or (not Z1) (and E2 Z1)) (or Z1 (not A2)) (or (not X3) (and N3 X3)) (or (not O2) (not (<= E 0))) (or (not O2) (and J2 O2)) (or O2 (not P2)) (or (not U1) (and U1 Z1)) (or (not K2) J2) (or (not B3) (and T2 B3)) (or (not Q) (and U Q)) (or Q (not R)) (or (not U) (not (<= E1 0))) (or (not U) (and U O2)) (or (not V) U) (or (not N3) (and N3 B3)) (or (not F2) E2) (or (not W3) (and W3 X3)) (or E3 (not N3)) (or (not V1) U1) (= S true) (= W1 true) (= W3 true) (= B2 M1)))) (main@NewDefault.outer.i Z3 A4 B4 C4 D4))))
(constraint (forall ((A Int) (B Int) (C Int) (D Int) (E Int) (F Int) (G Int) (H Int) (I Int) (J Int) (K Int) (L Bool) (M Int) (N Bool) (O Int) (P Bool) (Q Bool) (R Bool) (S Int) (T Bool) (U Bool) (V Bool) (W Int) (X Int) (Y Int) (Z Int) (A1 Int) (B1 Int) (C1 Int) (D1 Bool) (E1 Int) (F1 Bool) (G1 Int) (H1 Bool) (I1 Int) (J1 Int) (K1 Bool) (L1 Bool) (M1 Bool) (N1 Int) (O1 Int) (P1 Bool) (Q1 Int) (R1 Int) (S1 Bool) (T1 Bool) (U1 Bool) (V1 Int) (W1 Int) (X1 Bool) (Y1 Bool) (Z1 Bool) (A2 Int) (B2 Int) (C2 Bool) (D2 Bool) (E2 Bool) (F2 Int) (G2 Int) (H2 Bool) (I2 Bool) (J2 Bool) (K2 Int) (L2 Int) (M2 Bool) (N2 Bool) (O2 Bool) (P2 Int) (Q2 Bool) (R2 Int) (S2 Int) (T2 Bool) (U2 Int) (V2 Int) (W2 Int) (X2 Bool) (Y2 Bool) (Z2 Bool) (A3 Int) (B3 Int) (C3 Int) (D3 Int) (E3 Int) (F3 Bool) (G3 Int) (H3 Int) (I3 Int)) (=> (and (main@_bb14 B3 P2 N1 O1 P1 G3 H3 I3) (let ((a!1 (= Q2 (and (not (<= 8 U2)) (>= U2 0))))) (and (= F1 (not K1)) (= Z1 K1) (= X2 T2) a!1 (= H1 (= W 0)) (= O2 P1) (= V P1) (= L (not (<= I1 O1))) (= M1 (= E 0)) (= J2 P1) (= D1 (not (<= G1 J1))) (= L1 (= D 0)) (= E2 K1) (= W1 J1) (= G2 O1) (= B2 J1) (= G (+ 48 G3 (* 136 B1))) (= A1 (+ 64 G3 (* 136 B1))) (= G1 (+ 1 I1)) (= O M) (= A (+ H3 (* 8 P2))) (= X (+ 104 G3 (* 136 B1))) (= Y (+ 124 G3 (* 136 B1))) (= C (+ 12 D)) (= A3 U2) (= M (+ 1 O1)) (= V1 G1) (= J (+ 8 G3 (* 136 B1))) (= A2 I1) (= W2 S2) (= E1 (+ 1 J1)) (= F P2) (= I (+ G3 (* 136 B1))) (= L2 O1) (= H (+ 112 G3 (* 136 B1))) (= S O1) (= Z (+ 56 G3 (* 136 B1))) (= U2 (+ 1 P2)) (= V2 R2) (= Q1 G1) (= R1 E1) (= K (+ 32 G3 (* 136 B1))) (= C1 (+ 88 G3 (* 136 B1))) (= K2 N1) (= B D) (= F2 N1) (= B1 F) (= I1 (+ 1 N1)) (not (<= H3 0)) (or (not Z2) (and S1 T1) (and I2 H2) (and C2 D2) (and Y1 X1) (and M2 N2)) (or (not X1) (<= G3 0) (not (<= A1 0))) (or (not X1) (<= G3 0) (not (<= X 0))) (or (not X1) (<= G3 0) (not (<= Y 0))) (or (not X1) (<= G3 0) (not (<= Z 0))) (or (not X1) (<= G3 0) (not (<= C1 0))) (or (not M2) (<= D 0) (not (<= C 0))) (or (not M2) (not N2) (= T2 O2)) (or (not M2) (not N2) (= R2 K2)) (or (not M2) (not N2) (= S2 L2)) (or (not Y2) (not Z2) (= F3 X2)) (or (not Y2) (not Z2) (= C3 A3)) (or (not Y2) (not Z2) (= D3 V2)) (or (not Y2) (not Z2) (= E3 W2)) (or (not Y2) Q2 (not Z2)) (or (not T) (<= G3 0) (not (<= G 0))) (or (not T) (<= G3 0) (not (<= J 0))) (or (not T) (<= G3 0) (not (<= I 0))) (or (not T) (<= G3 0) (not (<= H 0))) (or (not T) (<= G3 0) (not (<= K 0))) (or (not T) (not U) (= K1 V)) (or (not T) (not U) (= J1 S)) (or (not T) (not U) (not N)) (or (not P) (not T) N) (or (not Y1) (not X1) (= T2 Z1)) (or (not Y1) (not X1) (= R2 V1)) (or (not Y1) (not X1) (= S2 W1)) (or (not Y1) (not F1) (not X1)) (or M1 (not M2) (not N2)) (or (not M1) (not T) (not M2)) (or (not C2) (and Q P) (and T U)) (or (not C2) (not H1) (not X1)) (or (not C2) (not D2) (= T2 E2)) (or (not C2) (not D2) (= R2 A2)) (or (not C2) (not D2) (= S2 B2)) (or (not C2) (not D2) H1) (or (not Q) (not P) (= K1 R)) (or (not Q) (not P) (= J1 O)) (or (not Q) (not P) L) (or (not I2) (not H2) (= T2 J2)) (or (not I2) (not H2) (= R2 F2)) (or (not I2) (not H2) (= S2 G2)) (or (not L1) (not H2) (not M2)) (or L1 (not I2) (not H2)) (or (not S1) F1 (not X1)) (or (not S1) (not T1) (= T2 U1)) (or (not S1) (not T1) (= R2 Q1)) (or (not S1) (not T1) (= S2 R1)) (or (not S1) D1 (not T1)) (or (<= H3 0) (not (<= A 0))) (or (not X1) (not (<= G3 0))) (or (not X1) (and C2 X1)) (or (not M2) (not (<= D 0))) (or (not M2) (and H2 M2)) (or M2 (not N2)) (or (not Y2) (and Y2 Z2)) (or (not T) (not (<= G3 0))) (or (not T) (and T M2)) (or T (not U)) (or (not P) (and P T)) (or (not Y1) X1) (or C2 (not D2)) (or (not Q) P) (or (not I2) H2) (or (not S1) (and S1 X1)) (or S1 (not T1)) (= U1 true) (= Y2 true) (= R true) (= N (not P1))))) (main@_bb14 B3 C3 D3 E3 F3 G3 H3 I3))))
(constraint (forall ((A Int) (B Int) (C Int) (D Int) (E Int) (F Bool) (G Bool) (H Int) (I Int) (J Int) (K Bool) (L Bool) (M Int) (N Int) (O Int) (P Int) (Q Int) (R Int) (S Int) (T Int) (U Bool) (V Int) (W Bool) (X Int) (Y Bool) (Z Int) (A1 Bool) (B1 Bool) (C1 Bool) (D1 Int) (E1 Bool) (F1 Bool) (G1 Bool) (H1 Int) (I1 Bool) (J1 Bool) (K1 Int) (L1 Int) (M1 Int) (N1 Int) (O1 Int) (P1 Int) (Q1 Int) (R1 Int) (S1 Bool) (T1 Bool) (U1 Bool) (V1 Int) (W1 Int) (X1 Int) (Y1 Bool) (Z1 Bool) (A2 Bool) (B2 Bool) (C2 Bool) (D2 Bool)) (=> (and (main@_bb14 A M P X Y R1 D B) (and (= U1 (not S1)) (= U (not (<= L1 X))) (= J1 (= H1 0)) (= G1 Y) (= Z1 (not (<= W1 X1))) (= L (= J 0)) (= G (= I 0)) (= R (+ R1 (* 136 P1))) (= M1 (+ 124 R1 (* 136 P1))) (= H (+ 12 I)) (= O (+ 48 R1 (* 136 P1))) (= S (+ 8 R1 (* 136 P1))) (= K1 (+ 104 R1 (* 136 P1))) (= V1 (+ 1 X1)) (= Q (+ 112 R1 (* 136 P1))) (= N1 (+ 56 R1 (* 136 P1))) (= E I) (= L1 (+ 1 P)) (= V (+ 1 X)) (= O1 (+ 64 R1 (* 136 P1))) (= N M) (= W1 (+ 1 L1)) (= Z V) (= C (+ D (* 8 M))) (= D1 X) (= P1 N) (= Q1 (+ 88 R1 (* 136 P1))) (= T (+ 32 R1 (* 136 P1))) (not (<= D 0)) (or (not I1) (and E1 F1) (and B1 A1)) (or (not T1) (<= R1 0) (not (<= M1 0))) (or (not T1) (<= R1 0) (not (<= K1 0))) (or (not T1) (<= R1 0) (not (<= N1 0))) (or (not T1) (not (<= O1 0)) (<= R1 0)) (or (not T1) (not (<= Q1 0)) (<= R1 0)) (or (not T1) (not Y1) U1) (or (not T1) (not J1) (not I1)) (or (not K) (<= I 0) (not (<= H 0))) (or (not B1) (not A1) (= S1 C1)) (or (not B1) (not A1) (= X1 Z)) (or (not B1) (not A1) U) (or (not Z1) (not A2) (not Y1)) (or (not C2) (and B2 C2) (and C2 A2)) (or (not B2) (not A1) (not U)) (or (not E1) (<= R1 0) (not (<= R 0))) (or (not E1) (<= R1 0) (not (<= O 0))) (or (not E1) (<= R1 0) (not (<= S 0))) (or (not E1) (<= R1 0) (not (<= Q 0))) (or (not E1) (not (<= T 0)) (<= R1 0)) (or (not E1) (not A1) W) (or (not E1) (not F1) (= S1 G1)) (or (not E1) (not F1) (= X1 D1)) (or (not E1) (not F1) (not W)) (or (not E1) (not L) (not K)) (or (not G) (not F) (not K)) (or (<= D 0) (not (<= C 0))) (or (not D2) (and C2 D2)) (or (not Y1) (and T1 Y1)) (or (not A1) (and E1 A1)) (or (not T1) (not (<= R1 0))) (or (not T1) (and T1 I1)) (or (not A2) (and A2 Y1)) (or (not K) (not (<= I 0))) (or (not K) (and F K)) (or (not B1) A1) (or (not B2) (and B2 A1)) (or (not E1) (not (<= R1 0))) (or (not E1) (and E1 K)) (or E1 (not F1)) (= D2 true) (= C1 true) (= W (not Y)))) main@verifier.error.split)))
(constraint (forall ((CHC_COMP_UNUSED Bool)) (=> (and main@verifier.error.split true) false)))

(check-synth)

