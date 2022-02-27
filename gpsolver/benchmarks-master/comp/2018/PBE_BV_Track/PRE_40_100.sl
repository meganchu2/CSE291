
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


(constraint (= (f #x7cac982a47748ee0) #x00003e564c1523bb))
(constraint (= (f #xeb20c92cd92a0726) #x0000759064966c96))
(constraint (= (f #x26a85358e05e0773) #x4d50a6b1c0bc0ee6))
(constraint (= (f #xa684be05e84bee84) #xcd097c0bd097dd08))
(constraint (= (f #xebce57c9eda76ca6) #x000075e72be4f6d4))
(constraint (= (f #x98bb8ae47eba8e69) #xb17715c8fd751cd2))
(constraint (= (f #x756d4ccc24ee74b2) #xeada999849dce964))
(constraint (= (f #x70574b3e6722b53a) #x0000382ba59f3392))
(constraint (= (f #xedc04644a52ccee7) #x000076e023225297))
(constraint (= (f #x346a2556e615b810) #x68d44aadcc2b7020))
(constraint (= (f #xeb7093e601c9eee4) #x000075b849f300e5))
(constraint (= (f #xc67728bb839e8c92) #x0000633b945dc1d0))
(constraint (= (f #x0da65c4be3e98c85) #x000006d32e25f1f5))
(constraint (= (f #x2e5b7e5c48948d18) #x5cb6fcb891291a30))
(constraint (= (f #x3a573d8bb6d2617e) #x74ae7b176da4c2fc))
(constraint (= (f #x71757b19d016d4b1) #xe2eaf633a02da962))
(constraint (= (f #xc5a42e0de757d24c) #x000062d21706f3ac))
(constraint (= (f #x10a5a16ae233cb92) #x214b42d5c4679724))
(constraint (= (f #x8ce3d5d47c90b7e6) #x99c7aba8f9216fcc))
(constraint (= (f #x54721568ba0458ae) #xa8e42ad17408b15c))
(constraint (= (f #x60da7a98dad7b168) #xc1b4f531b5af62d0))
(constraint (= (f #x9ae43d99b58c3d53) #x00004d721eccdac7))
(constraint (= (f #x26908c3e50e9284d) #x4d21187ca1d2509a))
(constraint (= (f #x0ea1e15ec1797ec7) #x00000750f0af60bd))
(constraint (= (f #xc1524647041b660a) #x02a48c8e0836cc14))
(constraint (= (f #xd6894850dd8aaeec) #x00006b44a4286ec6))
(constraint (= (f #xe3ee2cdc86be4970) #x47dc59b90d7c92e0))
(constraint (= (f #x9c3be201b6716e34) #xb877c4036ce2dc68))
(constraint (= (f #x87a14ee04d52e984) #x000043d0a77026aa))
(constraint (= (f #x579a9b56793ee3e1) #x00002bcd4dab3ca0))
(constraint (= (f #xcaaa1ee479ec9e93) #x000065550f723cf7))
(constraint (= (f #x5aee27aedbe54811) #x00002d7713d76df3))
(constraint (= (f #x95c2deb1e885b787) #xab85bd63d10b6f0e))
(constraint (= (f #xeee1a5bbe6180d22) #x5dc34b77cc301a44))
(constraint (= (f #xd49062148c76e9a6) #x2920c42918edd34c))
(constraint (= (f #x3ed587db57ce2e24) #x00001f6ac3edabe8))
(constraint (= (f #x3c21e8b20d4e4ba9) #x00001e10f45906a8))
(constraint (= (f #x218b95e376e91370) #x43172bc6edd226e0))
(constraint (= (f #xe34ee0c256660932) #x469dc184accc1264))
(constraint (= (f #x7ec4b7353635cd2a) #xfd896e6a6c6b9a54))
(constraint (= (f #x9e2637408085c771) #xbc4c6e81010b8ee2))
(constraint (= (f #xdbc19763e76901c1) #x00006de0cbb1f3b5))
(constraint (= (f #xad63acd30e14e22d) #xdac759a61c29c45a))
(constraint (= (f #x67e77189bce61e4b) #xcfcee31379cc3c96))
(constraint (= (f #x7b6c19681e85e2b7) #xf6d832d03d0bc56e))
(constraint (= (f #x19e3213b0b1adec8) #x00000cf1909d858e))
(constraint (= (f #xed20455719a5886e) #x0000769022ab8cd3))
(constraint (= (f #x6879360eeab453c8) #xd0f26c1dd568a790))
(constraint (= (f #xc75ac5476669edad) #x0eb58a8eccd3db5a))
(constraint (= (f #x032bac1c18216ec4) #x065758383042dd88))
(constraint (= (f #xeaeab93555d5c96e) #x000075755c9aaaeb))
(constraint (= (f #x1eadeedd68a02099) #x3d5bddbad1404132))
(constraint (= (f #x50d9ac8831c58a56) #x0000286cd64418e3))
(constraint (= (f #xeebe75e84eaee0e3) #x5d7cebd09d5dc1c6))
(constraint (= (f #x6b39268e94d66c98) #xd6724d1d29acd930))
(constraint (= (f #xe6e8bc7882a8a84b) #x4dd178f105515096))
(constraint (= (f #x92de084de2747eda) #xa5bc109bc4e8fdb4))
(constraint (= (f #x1e666b7181cd945d) #x00000f3335b8c0e7))
(constraint (= (f #x11ea476801dd30e7) #x000008f523b400ef))
(constraint (= (f #xde90e3088b828728) #x00006f48718445c2))
(constraint (= (f #x74641033e1a7cc9e) #x00003a320819f0d4))
(constraint (= (f #xacb4d57e23d25899) #x0000565a6abf11ea))
(constraint (= (f #x683912a24e06b108) #xd07225449c0d6210))
(constraint (= (f #x68a132e411481586) #x00003450997208a5))
(constraint (= (f #xe31a97301d940d5e) #x0000718d4b980ecb))
(constraint (= (f #xc198eb303d303db6) #x000060cc75981e99))
(constraint (= (f #xc8cdb2a16b1ecd57) #x00006466d950b590))
(constraint (= (f #xa15045400e930c78) #xc2a08a801d2618f0))
(constraint (= (f #xcd6ed4e040c2118e) #x1adda9c08184231c))
(constraint (= (f #x3c736dba5ae31eba) #x78e6db74b5c63d74))
(constraint (= (f #xc9785703c157e7e5) #x000064bc2b81e0ac))
(constraint (= (f #x07832781d45d07c5) #x0f064f03a8ba0f8a))
(constraint (= (f #xb78a95e0e1372266) #x00005bc54af0709c))
(constraint (= (f #xc82ac58a14e71c51) #x10558b1429ce38a2))
(constraint (= (f #x7896c73c87eeea0c) #x00003c4b639e43f8))
(constraint (= (f #x57b355594053c154) #xaf66aab280a782a8))
(constraint (= (f #xa40815cb96a1934b) #xc8102b972d432696))
(constraint (= (f #x790d540458eb6e84) #xf21aa808b1d6dd08))
(constraint (= (f #xea9a56713e892150) #x5534ace27d1242a0))
(constraint (= (f #x55c3e3ed09eb1cc3) #x00002ae1f1f684f6))
(constraint (= (f #x4eb5c5e2916206c0) #x0000275ae2f148b2))
(constraint (= (f #x552e715636470313) #xaa5ce2ac6c8e0626))
(constraint (= (f #xeaae906297eee892) #x0000755748314bf8))
(constraint (= (f #x2134ded8ee14ceb8) #x4269bdb1dc299d70))
(constraint (= (f #xe58ab4d961cc2090) #x000072c55a6cb0e7))
(constraint (= (f #xe35c290493c87023) #x000071ae148249e5))
(constraint (= (f #xcab03e3e6660ec54) #x15607c7cccc1d8a8))
(constraint (= (f #x0e8ae4db36a66234) #x1d15c9b66d4cc468))
(constraint (= (f #x552ec34498187b0b) #xaa5d86893030f616))
(constraint (= (f #xbae710b620227e72) #xf5ce216c4044fce4))
(constraint (= (f #x5287e22b5e87e49b) #xa50fc456bd0fc936))
(constraint (= (f #x69349e3e56208a66) #xd2693c7cac4114cc))
(constraint (= (f #xce7230299e7c1ac9) #x1ce460533cf83592))
(constraint (= (f #x87e0bc9310edcaec) #x8fc1792621db95d8))
(constraint (= (f #x55b2abe82bdbec7b) #x00002ad955f415ee))
(constraint (= (f #x18be5eb5574a3e8b) #x00000c5f2f5aaba6))
(constraint (= (f #x18e4c7809b88ebd0) #x00000c7263c04dc5))
(constraint (= (f #x9a485e73e5591859) #x00004d242f39f2ad))
(constraint (= (f #xe48044b5ca7ee71a) #x4900896b94fdce34))
(constraint (= (f #xa1a0b0eebea037be) #xc34161dd7d406f7c))
(check-synth)
