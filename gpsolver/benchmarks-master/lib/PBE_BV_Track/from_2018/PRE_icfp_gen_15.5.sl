(set-logic BV)

(define-fun ehad ((x (_ BitVec 64))) (_ BitVec 64)
    (bvlshr x #x0000000000000001))
(define-fun arba ((x (_ BitVec 64))) (_ BitVec 64)
    (bvlshr x #x0000000000000004))
(define-fun shesh ((x (_ BitVec 64))) (_ BitVec 64)
    (bvlshr x #x0000000000000010))
(define-fun smol ((x (_ BitVec 64))) (_ BitVec 64)
    (bvshl x #x0000000000000001))
(define-fun im ((x (_ BitVec 64)) (y (_ BitVec 64)) (z (_ BitVec 64))) (_ BitVec 64)
    (ite (= x #x0000000000000001) y z))
(synth-fun f ((x (_ BitVec 64))) (_ BitVec 64)
    ((Start (_ BitVec 64)))
    ((Start (_ BitVec 64) (#x0000000000000000 #x0000000000000001 x (bvnot Start) (smol Start) (ehad Start) (arba Start) (shesh Start) (bvand Start Start) (bvor Start Start) (bvxor Start Start) (bvadd Start Start) (im Start Start Start)))))

(constraint (= (f #xCB5E5719B3DA8985) #x0000000000000001))
(constraint (= (f #x9507EDBA7D6A2142) #x0000000000000000))
(constraint (= (f #x3C1AF83DC25EB51B) #x0000000000000001))
(constraint (= (f #x5D9073FB16EBE1C9) #x0000000000000001))
(constraint (= (f #x3716A561DF826A5F) #x0000000000000001))
(constraint (= (f #xFD6BFD2FBDFAFDE6) #x0000000000000000))
(constraint (= (f #xFC7E7F1ED6D6BF8E) #x0000000000000000))
(constraint (= (f #xFE5BFE1FCFDFD3E6) #x0000000000000000))
(constraint (= (f #xFFAF3DBD3F5B6968) #x0000000000000000))
(constraint (= (f #xF6B7F1F7FE1ED3C2) #x0000000000000000))
(constraint (= (f #xE2B3A988A4287B7E) #x0000000000000000))
(constraint (= (f #x0AE582DFA9A9F840) #x0000000000000000))
(constraint (= (f #xF1C6C468AD7022A7) #x0000000000000000))
(constraint (= (f #x2EF55D2A7395309F) #x0000000000000000))
(constraint (= (f #x4C72E36DEA81739F) #x0000000000000000))
(constraint (= (f #xFFFFFFFFFFFFFFFF) #x0000000000000000))
(constraint (= (f #xF3C3F2F696FDEF2E) #x0000000000000000))
(constraint (= (f #xF78FF7FC3CF1F7A6) #x0000000000000000))
(constraint (= (f #xF8787DEFDF6DA7AE) #x0000000000000000))
(constraint (= (f #xF4FAFA7DF2F96DA6) #x0000000000000000))
(constraint (= (f #xF6BFD2DA7CBDBF2C) #x0000000000000000))
(constraint (= (f #x608FFF4EE8BC2DFA) #x0000000000000000))
(constraint (= (f #x9075B01383C01D98) #x0000000000000000))
(constraint (= (f #xFDD0FC8CB6990762) #x0000000000000000))
(constraint (= (f #x3BC26347F52D0667) #x0000000000000000))
(constraint (= (f #x84CB178CC235378B) #x0000000000000000))
(constraint (= (f #x25958EA0E36B7486) #x0000000000000000))
(constraint (= (f #x8E9167626B020AAF) #x0000000000000001))
(constraint (= (f #x9849DCD4435A0CC2) #x0000000000000000))
(constraint (= (f #x4F3B7F080E72F36F) #x0000000000000001))
(constraint (= (f #x0DC43D58675AC120) #x0000000000000000))
(constraint (= (f #xF2F2D7E5BE7EDF0E) #x0000000000000000))
(constraint (= (f #xFFFFFFFFFFFFFFFF) #x0000000000000000))
(constraint (= (f #x7FFFFFFFFFFFFFFF) #x0000000000000001))
(constraint (= (f #x7069E10D94CBF767) #x0000000000000001))

(check-synth)

