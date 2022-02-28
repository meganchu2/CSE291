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

(constraint (= (f #xC09FB079EA232724) #x003E40E380040C00))
(constraint (= (f #x5EDCC0DE63C6E4A6) #x3931013887098008))
(constraint (= (f #x88C0481357958B0A) #x010000040E020400))
(constraint (= (f #xDB22EA0BDDD9062C) #x2401800733200810))
(constraint (= (f #x85706BE366E35BA7) #x00C087848984260C))
(constraint (= (f #x0000000000000001) #x0000000000000000))
(constraint (= (f #xFFFFFF0000010001) #x0000000000000000))
(constraint (= (f #x24D9525581BBAA0F) #x24D976DDD3FFABBF))
(constraint (= (f #x433E77AC77C52F6F) #x433E77BE77ED7FEF))
(constraint (= (f #x669B643CF6D45B0F) #x669B66BFF6FCFFDF))
(constraint (= (f #x246D991F6E832B2F) #x246DBD7FFF9F6FAF))
(constraint (= (f #xCEDCB3BF8894088F) #xCEDCFFFFBBBF889F))
(constraint (= (f #x1E131677C72EE79F) #x380408CF0C198E3C))
(constraint (= (f #x5418083671C31EBF) #x00200048C304387C))
(constraint (= (f #x735FA04F47524F9F) #xC43E001C0C001E3C))
(constraint (= (f #x713FFE164A85B67F) #xC07FF808000248FC))
(constraint (= (f #x4C7E537AADA3BBBF) #x10F804E01206667C))
(constraint (= (f #x000000000000000C) #x0000000000000010))
(constraint (= (f #x000000000000000E) #x0000000000000018))
(constraint (= (f #x000000000000000B) #x0000000000000004))
(constraint (= (f #x0000000000000009) #x0000000000000000))
(constraint (= (f #x000000000000000D) #x0000000000000010))
(constraint (= (f #x0000000000000000) #x0000000000000000))
(constraint (= (f #xD81032E56C01FFF3) #xD810FAF57EE5FFF3))
(constraint (= (f #x8F2FDF3860F48212) #x8F2FDF3FFFFCE2F6))
(constraint (= (f #x85F4872FE2A8917C) #x85F487FFE7AFF3FC))
(constraint (= (f #x2B39EE07855D0672) #x2B39EF3FEF5F877F))
(constraint (= (f #xFC270232F38A0B7D) #xFC27FE37F3BAFBFF))
(constraint (= (f #x000000000000000F) #x000000000000001C))
(constraint (= (f #xFFFFFFFFFFFFFFF3) #xFFFFFFFFFFFFFFFF))
(constraint (= (f #xFFFFFFFFFFFFFFF6) #xFFFFFFFFFFFFFFFF))
(constraint (= (f #xFFFFFFFFFFFFFFF7) #xFFFFFFFFFFFFFFFF))
(constraint (= (f #xFFFFFFFFFFFFFFF1) #xFFFFFFFFFFFFFFFF))
(constraint (= (f #xFFFFFFFFFFFFFFF0) #xFFFFFFFFFFFFFFFF))

(check-synth)
