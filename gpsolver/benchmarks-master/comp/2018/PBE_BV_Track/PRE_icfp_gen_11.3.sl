(set-logic BV)

(define-fun ehad ((x (BitVec 64))) (BitVec 64) (bvlshr x #x0000000000000001))
(define-fun arba ((x (BitVec 64))) (BitVec 64) (bvlshr x #x0000000000000004))
(define-fun shesh ((x (BitVec 64))) (BitVec 64) (bvlshr x #x0000000000000010))
(define-fun smol ((x (BitVec 64))) (BitVec 64) (bvshl x #x0000000000000001))
(define-fun im ((x (BitVec 64)) (y (BitVec 64)) (z (BitVec 64))) (BitVec 64) (ite (= x #x0000000000000001) y z))

(synth-fun f ( (x (BitVec 64))) (BitVec 64)
(

(Start (BitVec 64) (#x0000000000000000 #x0000000000000001 x (bvnot Start)
                    (smol Start)
 		    (ehad Start)
		    (arba Start)
		    (shesh Start)
		    (bvand Start Start)
		    (bvor Start Start)
		    (bvxor Start Start)
		    (bvadd Start Start)
		    (im Start Start Start)
 ))
)
)
(constraint (= (f #x95BAC00BBCB0FEFD) #xD48A7FE8869E0205))
(constraint (= (f #xF1A6A52A909CDBF3) #x1CB2B5AADEC64819))
(constraint (= (f #x1F9C279958E25E2F) #xC0C7B0CD4E3B43A1))
(constraint (= (f #x1D2DB8F3369062EB) #xC5A48E1992DF3A29))
(constraint (= (f #xD5CC22BA4DFF1345) #x5467BA8B6401D975))
(constraint (= (f #xD429A89C2546761A) #x0000000000000002))
(constraint (= (f #xDBD570FB401F57D8) #x0000000000000002))
(constraint (= (f #x63F012F727931EC0) #x0000000000000002))
(constraint (= (f #xA2E89AB664D2D1AC) #x0000000000000002))
(constraint (= (f #xFEA9C5D2125FFC3C) #x0000000000000002))
(constraint (= (f #x0000000000000016) #xFFFFFFFFFFFFFFD3))
(constraint (= (f #x5555555555555556) #x5555555555555553))
(constraint (= (f #x0000000000015556) #xFFFFFFFFFFFD5553))
(constraint (= (f #x0000000000000156) #xFFFFFFFFFFFFFD53))
(constraint (= (f #xE0FA0D68D9228020) #x0000000000000002))
(constraint (= (f #x1093B6A147A9C23F) #xDED892BD70AC7B81))
(constraint (= (f #x5FB51CC354498B15) #x4095C679576CE9D5))
(constraint (= (f #x0D0E80B218F91501) #xE5E2FE9BCE0DD5FD))
(constraint (= (f #x2182C4C4A0C8DB9E) #x0000000000000002))
(constraint (= (f #x7E36684DC561FC0E) #x0000000000000002))
(constraint (= (f #xF0E7A714049C3D83) #x1E30B1D7F6C784F9))
(constraint (= (f #x5BCEA175DD35FF04) #x0000000000000002))
(constraint (= (f #x1CEEC5FF6D087352) #x0000000000000002))
(constraint (= (f #x9CBD492140554126) #x0000000000000002))
(constraint (= (f #x5555555555555556) #x5555555555555553))
(constraint (= (f #xE2F79146EEDFC59A) #x0000000000000002))
(constraint (= (f #x545488108954A206) #x0000000000000002))
(constraint (= (f #x9129091448229056) #x0000000000000002))
(constraint (= (f #x52A5480481291116) #x0000000000000002))
(constraint (= (f #x0555555555555556) #xF555555555555553))
(constraint (= (f #x954125524A552096) #x0000000000000002))
(constraint (= (f #x8949248952514906) #x0000000000000002))
(constraint (= (f #x9F6E6EA9E8190DD6) #x0000000000000002))
(constraint (= (f #x22888941492A0516) #x0000000000000002))
(constraint (= (f #x540E28702A0C4232) #x0000000000000002))
(constraint (= (f #x38A8C01C040C071E) #x0000000000000002))
(constraint (= (f #x86606231131D1902) #x0000000000000002))
(constraint (= (f #x851290888090A556) #x0000000000000002))
(constraint (= (f #x02AAABBBBBBBBBBA) #x0000000000000002))
(constraint (= (f #xA05050C38541C552) #x0000000000000002))
(constraint (= (f #x4AA540A514954916) #x0000000000000002))
(constraint (= (f #xA8E0800438E1003A) #x0000000000000002))
(constraint (= (f #xA058300E88268076) #x0000000000000002))
(constraint (= (f #xD3809023904C2552) #x0000000000000002))
(constraint (= (f #x92408A1441054926) #x0000000000000002))
(constraint (= (f #x70A304441C544056) #x0000000000000002))
(constraint (= (f #x0454285142144516) #x0000000000000002))
(constraint (= (f #xA928248204514516) #x0000000000000002))
(constraint (= (f #x000000000011115E) #x0000000000000002))
(constraint (= (f #x0011111111111332) #x0000000000000002))
(constraint (= (f #xA8A20A8002A10A1A) #x0000000000000002))
(constraint (= (f #x4555555555555556) #x0000000000000002))
(constraint (= (f #x00000AAAAFFFFFFE) #x0000000000000002))
(constraint (= (f #x0000000000000002) #xFFFFFFFFFFFFFFFB))
(constraint (= (f #x0000000000000014) #x0000000000000002))
(constraint (= (f #x7776666666666646) #x0000000000000002))
(constraint (= (f #xCFDF4F524F504B02) #x0000000000000002))
(constraint (= (f #x0000000000000006) #xFFFFFFFFFFFFFFF3))
(constraint (= (f #x0000000000000000) #x0000000000000002))
(constraint (= (f #x1FFFFFFFFFFFFFFE) #x0000000000000002))
(check-synth)
