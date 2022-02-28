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

(constraint (= (f #x3369A72F176F1F84) #x3369A72F176F1F84))
(constraint (= (f #xC62A3884D3684C3C) #xC62A3884D3684C3C))
(constraint (= (f #x03EC5698A1FF105A) #x03EC5698A1FF105A))
(constraint (= (f #xCB0CF3CC3F416682) #xCB0CF3CC3F416682))
(constraint (= (f #xE27BD92AD47C51EE) #xE27BD92AD47C51EE))
(constraint (= (f #x4D76B08BDD6B8C20) #x135DAC22F75AE308))
(constraint (= (f #x860D88797EC3AB76) #x2183621E5FB0EADD))
(constraint (= (f #xF73E80B40259DCCE) #x3DCFA02D00967733))
(constraint (= (f #xF27CEA6D57DA9196) #x3C9F3A9B55F6A465))
(constraint (= (f #x2873E05771FFF5B0) #x0A1CF815DC7FFD6C))
(constraint (= (f #x143D5860451F749C) #x0000000000000001))
(constraint (= (f #x202961AD583537DC) #x0000000000000001))
(constraint (= (f #x83F21D3611186BA8) #x0000000000000001))
(constraint (= (f #xD7B6B1951FB40A8A) #x0000000000000001))
(constraint (= (f #xBCABE9A7A3245B6E) #x0000000000000001))
(constraint (= (f #x4D284CC80BD21241) #x4D284CC80BD21241))
(constraint (= (f #x397653B97D4F7133) #x397653B97D4F7133))
(constraint (= (f #x147D11DFFB6F2D87) #x147D11DFFB6F2D87))
(constraint (= (f #x2DB49001ABCD27B3) #x2DB49001ABCD27B3))
(constraint (= (f #x8E34914D85C07FA9) #x8E34914D85C07FA9))
(constraint (= (f #xC24D7377AC34E958) #x30935CDDEB0D3A56))
(constraint (= (f #xC8C58A193812D346) #x323162864E04B4D1))
(constraint (= (f #x8B62AB64E5088A46) #x22D8AAD939422291))
(constraint (= (f #xF78B70EF430BBF00) #x3DE2DC3BD0C2EFC0))
(constraint (= (f #xA10D3C1D3022FCC2) #x28434F074C08BF30))
(constraint (= (f #xF0F0F0F0F0F0F0F2) #x0000000000000000))
(constraint (= (f #xD4BF9E0715F4F013) #xD4BF9E0715F4F013))
(constraint (= (f #xFBBCB15662C5A35B) #xFBBCB15662C5A35B))
(constraint (= (f #x6C8D03C54B6FCA97) #x6C8D03C54B6FCA97))
(constraint (= (f #x7F62DAC954C9D729) #x7F62DAC954C9D729))
(constraint (= (f #x03895C71E94BE01F) #x03895C71E94BE01F))
(constraint (= (f #xB31382B4B51A3139) #x0000000000000001))
(constraint (= (f #xBE1915F487B34B21) #x0000000000000001))
(constraint (= (f #xF17B9E916087073B) #x0000000000000001))
(constraint (= (f #xBAD6F5FEDB2C0791) #x0000000000000001))
(constraint (= (f #x7E75CB7D0A3C3BDB) #x0000000000000001))
(constraint (= (f #xF8ED5D9F2926DBA1) #x0000000000000001))
(constraint (= (f #xE63DE1DE01B58653) #x0000000000000001))
(constraint (= (f #xCDCF09D64A31AF87) #x0000000000000001))
(constraint (= (f #xC3090C8DC13BF2AF) #x0000000000000001))
(constraint (= (f #xB194C91A7C8A96B5) #x0000000000000001))
(constraint (= (f #x0000000000000001) #x0000000000000001))

(check-synth)
