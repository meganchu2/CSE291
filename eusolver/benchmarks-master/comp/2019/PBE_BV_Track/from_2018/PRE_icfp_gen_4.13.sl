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

(constraint (= (f #x60FF5E56FEE68F6E) #x0307FAF2B7F7347B))
(constraint (= (f #x4BCDA3B95CA70344) #x025E6D1DCAE5381A))
(constraint (= (f #x056C648D2CFB7AC0) #x002B63246967DBD6))
(constraint (= (f #x26ECFF2929B34DAE) #x013767F9494D9A6D))
(constraint (= (f #xDB483BADDC8E8D8C) #x06DA41DD6EE4746C))
(constraint (= (f #xA1D6822670F92C47) #x5E297DD98F06D3B8))
(constraint (= (f #x2A4CDAFC49527B85) #xD5B32503B6AD847A))
(constraint (= (f #x28D9296303B1852B) #xD726D69CFC4E7AD4))
(constraint (= (f #xED9A5D26055B6301) #x1265A2D9FAA49CFE))
(constraint (= (f #xA05F34AF4CD2F1AD) #x5FA0CB50B32D0E52))
(constraint (= (f #xFFFFFFFFFFF1D14C) #x07FFFFFFFFFF8E8A))
(constraint (= (f #xFFFFFFFFFFF643A2) #x07FFFFFFFFFFB21D))
(constraint (= (f #xFFFFFFFFFFF67F2E) #x07FFFFFFFFFFB3F9))
(constraint (= (f #x7FFFFFFFFFF22882) #x03FFFFFFFFFF9144))
(constraint (= (f #x7FFFFFFFFFF2F0E8) #x03FFFFFFFFFF9787))
(constraint (= (f #x1D3752CA67E0AE58) #xE2C8AD35981F51A7))
(constraint (= (f #x9AB9D094B30060FA) #x65462F6B4CFF9F05))
(constraint (= (f #x25F342CDFE98061A) #xDA0CBD320167F9E5))
(constraint (= (f #x5873F6D107A819D4) #xA78C092EF857E62B))
(constraint (= (f #xCFBAE32037F76A9E) #x30451CDFC8089561))
(constraint (= (f #x7FFFFFFFFFF43DE3) #xFFFFFFFFFFFFFFFF))
(constraint (= (f #xFFFFFFFFFFF3F561) #xFFFFFFFFFFFFFFFF))
(constraint (= (f #xFFFFFFFFFFF5418F) #xFFFFFFFFFFFFFFFF))
(constraint (= (f #x7FFFFFFFFFF458AB) #xFFFFFFFFFFFFFFFF))
(constraint (= (f #xFFFFFFFFFFF50E6F) #xFFFFFFFFFFFFFFFF))
(constraint (= (f #xF800000000000001) #x07FFFFFFFFFFFFFE))
(constraint (= (f #x4A5C209ABF1DF0D5) #xB5A3DF6540E20F2A))
(constraint (= (f #x9951354B31499513) #x66AECAB4CEB66AEC))
(constraint (= (f #x765B2A4D5C178039) #x89A4D5B2A3E87FC6))
(constraint (= (f #xFCE990D756EA7A19) #x03166F28A91585E6))
(constraint (= (f #x53098C10CA93D6F1) #xACF673EF356C290E))
(constraint (= (f #x7FFFFFFFFFF03DDA) #xFFFFFFFFFFFFFFFF))
(constraint (= (f #xFFFFFFFFFFF0AE9C) #xFFFFFFFFFFFFFFFF))
(constraint (= (f #xFFFFFFFFFFF44D32) #xFFFFFFFFFFFFFFFF))
(constraint (= (f #xFFFFFFFFFFF4A05E) #xFFFFFFFFFFFFFFFF))
(constraint (= (f #x7FFFFFFFFFF5C4F6) #xFFFFFFFFFFFFFFFF))
(constraint (= (f #x7FFFFFFFFFF0C3D9) #xFFFFFFFFFFFFFFFF))
(constraint (= (f #xFFFFFFFFFFF55399) #xFFFFFFFFFFFFFFFF))
(constraint (= (f #xFFFFFFFFFFF35033) #xFFFFFFFFFFFFFFFF))
(constraint (= (f #x7FFFFFFFFFF474BF) #xFFFFFFFFFFFFFFFF))
(constraint (= (f #x7FFFFFFFFFF663D9) #xFFFFFFFFFFFFFFFF))
(constraint (= (f #xBD2E7EA9B463CE62) #x05E973F54DA31E73))
(constraint (= (f #x3E9D2A58375E0CEC) #x01F4E952C1BAF067))
(constraint (= (f #x549693D347DB136E) #x02A4B49E9A3ED89B))
(constraint (= (f #xA51D9C807D697B37) #x5AE2637F829684C8))
(constraint (= (f #xACBC0945B57EEDD9) #x5343F6BA4A811226))
(constraint (= (f #xCDE4C5943E78C87B) #x321B3A6BC1873784))
(constraint (= (f #x31AB192CE1EB4CA4) #x018D58C9670F5A65))
(constraint (= (f #xE14E457EC37821DA) #x1EB1BA813C87DE25))
(constraint (= (f #xB64A18144E5D70D8) #x49B5E7EBB1A28F27))
(constraint (= (f #xD1A7D7C2C6C511AD) #x2E58283D393AEE52))
(constraint (= (f #x7FFFFFFFFFF5C79F) #xFFFFFFFFFFFFFFFF))
(constraint (= (f #x0000000000000001) #xFFFFFFFFFFFFFFFE))
(constraint (= (f #x7FFFFFFFFFF7DF06) #x03FFFFFFFFFFBEF8))
(constraint (= (f #xFFFFFFFFFFF36C8B) #xFFFFFFFFFFFFFFFF))
(constraint (= (f #x7FFFFFFFFFF3055C) #xFFFFFFFFFFFFFFFF))
(constraint (= (f #xA808C1BE1012C52B) #x57F73E41EFED3AD4))
(constraint (= (f #x5C1C461BF3FF0667) #xA3E3B9E40C00F998))
(constraint (= (f #xACD8A2529C0B9A72) #x53275DAD63F4658D))
(constraint (= (f #xFFFFFFFFFFF1EB2C) #x07FFFFFFFFFF8F59))

(check-synth)

