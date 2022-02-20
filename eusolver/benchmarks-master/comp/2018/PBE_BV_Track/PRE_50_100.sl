
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


(constraint (= (f #x1ccd32c0d3ee03dc) #x1ccd32c0d3ee03dd))
(constraint (= (f #x46e0c8eedb1b7799) #x46e0c8eedb1b779a))
(constraint (= (f #x9690229d8d3a3eaa) #x9690229d8d3a3eab))
(constraint (= (f #x477e9473ac3e2690) #x477e9473ac3e2691))
(constraint (= (f #xe72be9b7ed79b576) #x0297c3ad8378adf9))
(constraint (= (f #x9c04272e280e723e) #x9c04272e280e723f))
(constraint (= (f #xce9e52378acc6a94) #xce9e52378acc6a95))
(constraint (= (f #x97ba217589e80db0) #x97ba217589e80db1))
(constraint (= (f #x9012d6cda9dbe355) #x0b0377b56fa6c25f))
(constraint (= (f #x942db4cbeac9eb38) #x0bc76dd5c3f5a3d4))
(constraint (= (f #x655e3eedde53507a) #x655e3eedde53507b))
(constraint (= (f #xc035a47646d92edb) #xc035a47646d92edc))
(constraint (= (f #xe5aae5e071e731e4) #xe5aae5e071e731e5))
(constraint (= (f #x931c2c50012076ad) #x931c2c50012076ae))
(constraint (= (f #xb98dae0de379e94a) #x0ca96f216258a3bd))
(constraint (= (f #x1999a472e64882ce) #x02aaaec972ad9875))
(constraint (= (f #x1c01ede1e809dae2) #x024023622381a6f2))
(constraint (= (f #xe33a5c850eebe051) #x0254ee58f133c20f))
(constraint (= (f #x02bc60d069db370e) #x02bc60d069db370f))
(constraint (= (f #xed84c8dbc78e4990) #xed84c8dbc78e4991))
(constraint (= (f #xb87c9e27de32aabd) #x0c885a2686257ffc))
(constraint (= (f #x076e2b3d8a6ed70a) #x009b27d469eb3791))
(constraint (= (f #x5e6708eea160a0be) #x0e2a91933e3a1e1c))
(constraint (= (f #x9a7cb9de2d718cc6) #x0ae85ca627792954))
(constraint (= (f #x066324154a5b3a8a) #x066324154a5b3a8b))
(constraint (= (f #x94ec4ce192dcde66) #x0bd34d522b76562a))
(constraint (= (f #x73eea4a4ee04e329) #x09433eded320d257))
(constraint (= (f #xed1c028db5c68680) #x037240796de4b8b8))
(constraint (= (f #x0d9c940155340b22) #x0d9c940155340b23))
(constraint (= (f #x8061cd356be60d89) #x8061cd356be60d8a))
(constraint (= (f #xe78e1d6d3eee1ae3) #xe78e1d6d3eee1ae4))
(constraint (= (f #xd7ed79eacba36743) #xd7ed79eacba36744))
(constraint (= (f #x18cc109d289643c3) #x18cc109d289643c4))
(constraint (= (f #xb6e91d8be63a03d8) #xb6e91d8be63a03d9))
(constraint (= (f #x4eee76272c5deea1) #x0d3329a6974e633e))
(constraint (= (f #x1597ba2d2ee67877) #x1597ba2d2ee67878))
(constraint (= (f #x239442403abce991) #x064bcc6c04fc53ab))
(constraint (= (f #x9c263a22d54a7c89) #x9c263a22d54a7c8a))
(constraint (= (f #x21e7d11274345eca) #x21e7d11274345ecb))
(constraint (= (f #xe777c32c57b72d33) #xe777c32c57b72d34))
(constraint (= (f #xac954622995c15bd) #xac954622995c15be))
(constraint (= (f #x3d70d8e204d13345) #x3d70d8e204d13346))
(constraint (= (f #x9cb8c00345ed9371) #x0a5c94005ce36b59))
(constraint (= (f #xaae21928627eb5d3) #x0ff262b78a683de7))
(constraint (= (f #x7eb50b3a4bade96d) #x083df1d4edcf63bb))
(constraint (= (f #x4b307e97d5ebe3ae) #x0dd5083b87e3c24f))
(constraint (= (f #x948b5cec47bb14a2) #x948b5cec47bb14a3))
(constraint (= (f #xbccb5e3bd9450663) #xbccb5e3bd9450664))
(constraint (= (f #x4abced4ee6923238) #x4abced4ee6923239))
(constraint (= (f #xeee849c10ebeac8a) #x03338da4313c3f59))
(constraint (= (f #xb981eec4ed7ad0ed) #x0ca82334d378f713))
(constraint (= (f #x2045dedea00983a8) #x060ce6363e01a84f))
(constraint (= (f #x4642202dadcb117b) #x4642202dadcb117c))
(constraint (= (f #x3a610be36988c731) #x04ea31c25ba99495))
(constraint (= (f #xd44e0e9c902bed49) #x07cd213a5b07c37d))
(constraint (= (f #xeb9320997473c8c4) #x03cb561ab9c94594))
(constraint (= (f #x206363d0ca5b5c06) #x206363d0ca5b5c07))
(constraint (= (f #xb5cc3384b65045e4) #xb5cc3384b65045e5))
(constraint (= (f #x6919bc39a47713e7) #x6919bc39a47713e8))
(constraint (= (f #xc8b368cb7cb90873) #xc8b368cb7cb90874))
(constraint (= (f #xe7486beece9adeb4) #x029d8bc3353af63d))
(constraint (= (f #xa952e53e8ec202ce) #xa952e53e8ec202cf))
(constraint (= (f #x0a2e02aa784062eb) #x0a2e02aa784062ec))
(constraint (= (f #x303e15489ede5b58) #x303e15489ede5b59))
(constraint (= (f #xb86ee4336cc53d71) #xb86ee4336cc53d72))
(constraint (= (f #xb8e081499e2b0594) #xb8e081499e2b0595))
(constraint (= (f #xe6bb7edcd91164b9) #xe6bb7edcd91164ba))
(constraint (= (f #x73c2bd03de9da0e2) #x09447c70463a6e12))
(constraint (= (f #x23dc53d0a2c5731d) #x23dc53d0a2c5731e))
(constraint (= (f #xe249ea7ad0b3acb4) #x026da3e8f71d4f5d))
(constraint (= (f #x6e65d81cecd507d5) #x6e65d81cecd507d6))
(constraint (= (f #x2258e92659e4692b) #x2258e92659e4692c))
(constraint (= (f #xaa512354b445e555) #x0fef365fddcce2ff))
(constraint (= (f #x8c29e56eeee040d7) #x8c29e56eeee040d8))
(constraint (= (f #x119d07d3c10e477e) #x119d07d3c10e477f))
(constraint (= (f #x862200c426e1c7b5) #x08a66014c6b2248d))
(constraint (= (f #x4665c2a43dea738b) #x4665c2a43dea738c))
(constraint (= (f #x7c0e47b21e4e0cca) #x7c0e47b21e4e0ccb))
(constraint (= (f #xc5521ceb12065e6e) #xc5521ceb12065e6f))
(constraint (= (f #xd3e750a96498d1a2) #x07429f1fbada972e))
(constraint (= (f #x246ea75506c43e35) #x246ea75506c43e36))
(constraint (= (f #xdc299442e5401326) #xdc299442e5401327))
(constraint (= (f #x7aeeee1eb0e020b6) #x7aeeee1eb0e020b7))
(constraint (= (f #x5d330969e865eec9) #x0e7551bba38ae335))
(constraint (= (f #x3ae91db81dde91b5) #x04f3b26c82663b2d))
(constraint (= (f #xe9203481d072151e) #xe9203481d072151f))
(constraint (= (f #x46a2c5c942683dab) #x46a2c5c942683dac))
(constraint (= (f #x9e10b0853bbe49ec) #x9e10b0853bbe49ed))
(constraint (= (f #x62938c3e68271578) #x62938c3e68271579))
(constraint (= (f #x3a39d36e71e15d29) #x3a39d36e71e15d2a))
(constraint (= (f #x752ec82d4e5d71ad) #x752ec82d4e5d71ae))
(constraint (= (f #x240de48153c50ea9) #x240de48153c50eaa))
(constraint (= (f #x8505d6e3dad5070c) #x8505d6e3dad5070d))
(constraint (= (f #xcb8b8ebd271d17e4) #xcb8b8ebd271d17e5))
(constraint (= (f #x8cded49d9720bea4) #x095637da6b961c3e))
(constraint (= (f #x5d88ced7d2e2a76e) #x0e69953787727e9b))
(constraint (= (f #x0952e81ed29392ce) #x01bf7382377b4b75))
(constraint (= (f #xce98778124ed97ce) #x053a898836d36b85))
(constraint (= (f #xe4b4249836b0d75c) #x02ddc6da85bd179e))
(constraint (= (f #xa7bd861e7dea976a) #x0e8c68a22863fb9b))
(check-synth)
