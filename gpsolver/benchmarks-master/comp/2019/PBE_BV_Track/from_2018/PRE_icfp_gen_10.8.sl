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

(constraint (= (f #xDF9138D9B4009A94) #x40DD8E4C97FECAD4))
(constraint (= (f #x3C62F2850ADB9C58) #x873A1AF5EA48C74C))
(constraint (= (f #xC966539171686692) #x6D3358DD1D2F32D8))
(constraint (= (f #xE28817FE938ACA9A) #x3AEFD002D8EA6AC8))
(constraint (= (f #x46F6F4E96951FA48) #x7212162D2D5C0B6C))
(constraint (= (f #x5454E635B4BB4B16) #xABAB19CA4B44B4E9))
(constraint (= (f #xD7ABCED20DEAE5D8) #x2854312DF2151A27))
(constraint (= (f #xF15CFF960ABC63C0) #x0EA30069F5439C3F))
(constraint (= (f #x5A4B845FADF9BFBC) #xA5B47BA052064043))
(constraint (= (f #x6CBB433043CC7FF8) #x9344BCCFBC338007))
(constraint (= (f #x0000000000000032) #xFFFFFFFFFFFFFF98))
(constraint (= (f #x000000000000002A) #xFFFFFFFFFFFFFFA8))
(constraint (= (f #x0000000000000030) #xFFFFFFFFFFFFFF9C))
(constraint (= (f #x0000000000000036) #xFFFFFFFFFFFFFF90))
(constraint (= (f #x869098DBDDFE5943) #x796F67242201A6BC))
(constraint (= (f #x810A7E23555B3B49) #x7EF581DCAAA4C4B6))
(constraint (= (f #x660FD401095F17E5) #x99F02BFEF6A0E81A))
(constraint (= (f #xEF0C857169C748AD) #x10F37A8E9638B752))
(constraint (= (f #x30627F295930D871) #xCF9D80D6A6CF278E))
(constraint (= (f #x0000000000000031) #xFFFFFFFFFFFFFFCE))
(constraint (= (f #x0000000000000039) #xFFFFFFFFFFFFFFC6))
(constraint (= (f #x0000000000000023) #xFFFFFFFFFFFFFFDC))
(constraint (= (f #x000000000000003B) #xFFFFFFFFFFFFFFC4))
(constraint (= (f #x000000000000003D) #xFFFFFFFFFFFFFFC2))
(constraint (= (f #x1FFFFFFFFFFFFFFF) #xE000000000000000))
(constraint (= (f #x8A10B9FE40DCDB04) #x75EF4601BF2324FB))
(constraint (= (f #xE1A6C1B17164CEB6) #x3CB27C9D1D366290))
(constraint (= (f #x28DB6E40ACAEE09B) #xD72491BF53511F64))
(constraint (= (f #x7313AE4A6A79F1A1) #x8CEC51B595860E5E))
(constraint (= (f #x1DC77402627616D0) #xC47117FB3B13D25C))
(constraint (= (f #x45EB93DEB8F90888) #x7428D8428E0DEEEC))
(constraint (= (f #x105AB4994E202319) #xEFA54B66B1DFDCE6))
(constraint (= (f #x8A57F2CDC7DD948B) #x75A80D3238226B74))
(constraint (= (f #x604B42B3C89B0A14) #x3F697A986EC9EBD4))
(constraint (= (f #xFC74BFB6B8CFEA7D) #x038B404947301582))
(constraint (= (f #x0000000000000037) #xFFFFFFFFFFFFFFC8))
(constraint (= (f #x582C0ED1E3FEB82C) #x4FA7E25C38028FA4))
(constraint (= (f #x0000000000000030) #xFFFFFFFFFFFFFF9C))
(constraint (= (f #x71183D87C10087FA) #x8EE7C2783EFF7805))
(constraint (= (f #xF1BDB8C56DEB69CC) #x0E42473A92149633))
(constraint (= (f #x769E07E0AC1B2ECE) #x12C3F03EA7C9A260))
(constraint (= (f #x1FFFFFFFFFFFFFFF) #xE000000000000000))

(check-synth)

