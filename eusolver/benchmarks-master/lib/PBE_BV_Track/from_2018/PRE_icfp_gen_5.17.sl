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

(constraint (= (f #x1DB0D6A47A807B97) #x0000000000000002))
(constraint (= (f #xD20A5FA394B00D35) #x0000000000000002))
(constraint (= (f #x854EE008DC98CE4C) #x0000000000000000))
(constraint (= (f #xF3F8B3EE09A85143) #x0000000000000002))
(constraint (= (f #xD625BB933DBE2DE4) #x0000000000000000))
(constraint (= (f #x000000000000000A) #x0000000000000014))
(constraint (= (f #x0000000000000009) #x0000000000000012))
(constraint (= (f #x000000000000000F) #x000000000000001E))
(constraint (= (f #x6E1AFCE8DC69A315) #x4DD0FAC69B451AC0))
(constraint (= (f #x3EFB10A5954165AE) #xBCF2CE11403C510D))
(constraint (= (f #x2BDF2F82345108D1) #x839E8F79A30CE68C))
(constraint (= (f #xA5084BAD39C31509) #x10E72308B5BAC0E4))
(constraint (= (f #x3E86E6E1B7A57B9F) #xBC74D4DD2710735E))
(constraint (= (f #xECE210219274E491) #x0000000000000002))
(constraint (= (f #x4008203C65F401A1) #x0000000000000002))
(constraint (= (f #xA918002A4554530B) #x0000000000000002))
(constraint (= (f #x6A4D600B09DC0D63) #x0000000000000002))
(constraint (= (f #x985840302214245B) #x0000000000000002))
(constraint (= (f #x4490948150202913) #x324E427C0F9F84CA))
(constraint (= (f #x12892828814A908B) #xC86487867C204E62))
(constraint (= (f #x224915042924494B) #x9924C0F384932422))
(constraint (= (f #x528A9152A4A8954B) #x08604C0812064022))
(constraint (= (f #x24894914222A4923) #x926424C39981249A))
(constraint (= (f #x000000000000000B) #xFFFFFFFFFFFFFFE2))
(constraint (= (f #xA741B000201BD8C5) #xFFFFFFFFFFFFFFFC))
(constraint (= (f #x5266CA49005B0781) #xFFFFFFFFFFFFFFFC))
(constraint (= (f #x92F576402013D569) #xFFFFFFFFFFFFFFFC))
(constraint (= (f #xD28172C36053C969) #xFFFFFFFFFFFFFFFC))
(constraint (= (f #x335808C06811C82D) #xFFFFFFFFFFFFFFFC))
(constraint (= (f #x5094948AA4492083) #x0E42426013249E7A))
(constraint (= (f #x54AA512448891223) #x02010C932664C99A))
(constraint (= (f #x4A15554542A90A83) #x21C000303804E07A))
(constraint (= (f #xA9490A5524A12823) #x0424E100921C879A))
(constraint (= (f #xA91551280555044B) #x04C00C87F000F322))
(constraint (= (f #x802000894A520913) #xFFFFFFFFFFFFFFF8))
(constraint (= (f #x5412828A44924A83) #xFFFFFFFFFFFFFFF8))
(constraint (= (f #x5515120402942A83) #xFFFFFFFFFFFFFFF8))
(constraint (= (f #x2A52400A49521123) #xFFFFFFFFFFFFFFF8))
(constraint (= (f #x9290A442451092AB) #xFFFFFFFFFFFFFFF8))
(constraint (= (f #x15552AA80155254B) #xFFFFFFFFFFFFFFF8))
(constraint (= (f #x0550A420A0112513) #xFFFFFFFFFFFFFFF8))
(constraint (= (f #x21145444A8911013) #xFFFFFFFFFFFFFFF8))
(constraint (= (f #x511520288115212B) #xFFFFFFFFFFFFFFF8))
(constraint (= (f #x50828AA48255124B) #xFFFFFFFFFFFFFFF8))

(check-synth)

