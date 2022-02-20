(set-logic BV)

(define-fun ehad ((x (BitVec 64))) (BitVec 64)
    (bvlshr x #x0000000000000001))
(define-fun arba ((x (BitVec 64))) (BitVec 64)
    (bvlshr x #x0000000000000004))
(define-fun shesh ((x (BitVec 64))) (BitVec 64)
    (bvlshr x #x0000000000000010))
(define-fun smol ((x (BitVec 64))) (BitVec 64)
    (bvshl x #x0000000000000001))
(define-fun im ((x (BitVec 64)) (y (BitVec 64)) (z (BitVec 64))) (BitVec 64)
    (ite (= x #x0000000000000001) y z))
(synth-fun f ((x (BitVec 64))) (BitVec 64)
    ((Start (BitVec 64) (#x0000000000000000 #x0000000000000001 x (bvnot Start) (smol Start) (ehad Start) (arba Start) (shesh Start) (bvand Start Start) (bvor Start Start) (bvxor Start Start) (bvadd Start Start) (im Start Start Start)))))

(constraint (= (f #xFAF1F1CB040C5EB0) #x050E0E34FBF3A14F))
(constraint (= (f #x101B6656E864B2BC) #xEFE499A9179B4D43))
(constraint (= (f #xD2B2B403FE0733FE) #x2D4D4BFC01F8CC01))
(constraint (= (f #x07D8406BB0FC7D32) #xF827BF944F0382CD))
(constraint (= (f #x9D2B1F197D3678BC) #x62D4E0E682C98743))
(constraint (= (f #x0000000000000000) #x0000000000000000))
(constraint (= (f #x000000000000002A) #xFFFFFFFFFFFFFFAB))
(constraint (= (f #x0000000000000024) #xFFFFFFFFFFFFFFB7))
(constraint (= (f #x000000000000003C) #xFFFFFFFFFFFFFF87))
(constraint (= (f #x000000000000003A) #xFFFFFFFFFFFFFF8B))
(constraint (= (f #xA412071B6CC62515) #xA412071B6CC62517))
(constraint (= (f #x6B04EBD3231ABD7B) #x6B04EBD3231ABD7D))
(constraint (= (f #xEF3C32B87951374F) #xEF3C32B879513751))
(constraint (= (f #xCF215943B666B5EB) #xCF215943B666B5ED))
(constraint (= (f #x09B922804F091087) #x09B922804F091089))
(constraint (= (f #x0000000000000003) #x0000000000000000))
(constraint (= (f #x000000000000002B) #x000000000000002D))
(constraint (= (f #x000000000000002F) #x0000000000000031))
(constraint (= (f #x0000000000000035) #x0000000000000037))
(constraint (= (f #x0000000000000027) #x0000000000000029))
(constraint (= (f #xB6CC100489819014) #x4933EFFB767E6FEB))
(constraint (= (f #xB6869AC50459010F) #xB6869AC504590111))
(constraint (= (f #x18A5B5F58DC637C8) #xE75A4A0A7239C837))
(constraint (= (f #xA071CC02997D17D6) #x5F8E33FD6682E829))
(constraint (= (f #x735829C8A3963A74) #x8CA7D6375C69C58B))
(constraint (= (f #x9A87ACAA0A186F79) #x9A87ACAA0A186F7B))
(constraint (= (f #x6080F814FA08816D) #x6080F814FA08816F))
(constraint (= (f #x094A1E13E2246274) #xF6B5E1EC1DDB9D8B))
(constraint (= (f #xA369807134047A8E) #x5C967F8ECBFB8571))
(constraint (= (f #xF7E710F79B050487) #xF7E710F79B050489))
(constraint (= (f #x0000000000000003) #x0000000000000000))
(constraint (= (f #x0000000000000000) #x0000000000000000))
(constraint (= (f #x0000000000000032) #xFFFFFFFFFFFFFF9B))
(constraint (= (f #x000000000000003F) #x0000000000000041))
(constraint (= (f #x0000000000000001) #x0000000000000003))

(check-synth)

