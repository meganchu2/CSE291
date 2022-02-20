
(set-logic BV)

(define-fun shr1 ((x (BitVec 64))) (BitVec 64) (bvlshr x #x0000000000000001))
(define-fun shr4 ((x (BitVec 64))) (BitVec 64) (bvlshr x #x0000000000000004))
(define-fun shr16 ((x (BitVec 64))) (BitVec 64) (bvlshr x #x0000000000000010))
(define-fun shl1 ((x (BitVec 64))) (BitVec 64) (bvshl x #x0000000000000001))
(define-fun if0 ((x (BitVec 64)) (y (BitVec 64)) (z (BitVec 64))) (BitVec 64) (ite (= x #x0000000000000001) y z))

(synth-fun f ( (x (BitVec 64))) (BitVec 64)
(

(Start (BitVec 64) (#x0000000000000000 #x0000000000000001 x (bvnot Start)
                    (shl1 Start)
 		    (shr1 Start)
		    (shr4 Start)
		    (shr16 Start)
		    (bvand Start Start)
		    (bvor Start Start)
		    (bvxor Start Start)
		    (bvadd Start Start)
		    (if0 Start Start Start)
 ))
)
)


(constraint (= (f #x9b03ee8ccaab3ac0) #x026c0fba332aaceb))
(constraint (= (f #x5913aed87338242d) #x01644ebb61cce090))
(constraint (= (f #x4895ee5aadb70485) #x012257b96ab6dc12))
(constraint (= (f #xc7c52582e506a3d5) #x0484f6e872f0be47))
(constraint (= (f #x661ba57ddca8933b) #x0aa2cef8665f9b54))
(constraint (= (f #x1dd4867e8ce14e58) #x0267d8a839523d2e))
(constraint (= (f #x24d08a9b480cece6) #x0093422a6d2033b3))
(constraint (= (f #x6b3da74c83dd56c9) #x01acf69d320f755b))
(constraint (= (f #x2298a18bd4851772) #x067a9e29c7d8f399))
(constraint (= (f #x6622e76d64d5413e) #x0aa6729b7ad7fc34))
(constraint (= (f #xd0047919bce0c264) #x034011e466f38309))
(constraint (= (f #xb60d756e944e6187) #x02d835d5ba513986))
(constraint (= (f #x6ceb31c365170420) #x01b3acc70d945c10))
(constraint (= (f #x10260aec647e5335) #x0306a1f34ac82f55))
(constraint (= (f #x7618e431de578047) #x01d86390c7795e01))
(constraint (= (f #x6cba0eb4d0b446b4) #x0b5ce13dd71dccbd))
(constraint (= (f #xae58018366ce3650) #x0f2e80285ab525af))
(constraint (= (f #x1e5367024d115ece) #x00794d9c0934457b))
(constraint (= (f #xa886aee4be335333) #x0f98bf32dc255f55))
(constraint (= (f #x5c4a44e53de70488) #x0171291394f79c12))
(constraint (= (f #xe15d4089b0318c3a) #x023e7c19ad052944))
(constraint (= (f #x9e772b7b4cadca1e) #x0a2997d8dd5f65e2))
(constraint (= (f #x1aee0bd43254a34d) #x006bb82f50c9528d))
(constraint (= (f #xe8288c657ed15a32) #x0387994af8373ee5))
(constraint (= (f #x16d42de5ab8ee8aa) #x005b50b796ae3ba2))
(constraint (= (f #x6e5277b07bed3375) #x0b2f698d08c37559))
(constraint (= (f #x0e715528e3750b92) #x01293ff79259f1cb))
(constraint (= (f #xa89a94253bb0c771) #x0f9afbc6f4cd1499))
(constraint (= (f #x0637564ba4a4b114) #x00a59fadcededd33))
(constraint (= (f #x6053b0dcec23ca7d) #x0a0f4d16534645e8))
(constraint (= (f #x4125b68450e1423d) #x0c36edb8cf123c64))
(constraint (= (f #x68e6aeabcbeed8a3) #x01a39abaaf2fbb62))
(constraint (= (f #x4694c4801a33ab4b) #x011a53120068cead))
(constraint (= (f #x73636dd26c1c7bde) #x095a5b676b4248c6))
(constraint (= (f #x745c7eacab083cd5) #x09ce483f5fd18457))
(constraint (= (f #x503dc013cdee26e1) #x0140f7004f37b89b))
(constraint (= (f #xbcead51e5b759792) #x0c53f7f22ed9eb8b))
(constraint (= (f #x913eea97b8db4022) #x0244fbaa5ee36d00))
(constraint (= (f #x663c7b03041d16d4) #x0aa448d050c273b7))
(constraint (= (f #x12b0a0e16b95dbc5) #x004ac28385ae576f))
(constraint (= (f #xe2cdda6c5a53bc2e) #x038b3769b1694ef0))
(constraint (= (f #x0d457b55e0526c15) #x017cf8dfe20f6b43))
(constraint (= (f #xa2cce9c1c01b2a6b) #x028b33a707006ca9))
(constraint (= (f #xb0c3657e82ba3e5e) #x0d145af8387ce42e))
(constraint (= (f #xa1619521b75bc511) #x0e3a2bf62d9ec4f3))
(constraint (= (f #x8e2e140bab1c9ba6) #x0238b8502eac726e))
(constraint (= (f #x5e7dee4c457dba48) #x0179f7b93115f6e9))
(constraint (= (f #xd67997a3e12c740e) #x0359e65e8f84b1d0))
(constraint (= (f #x57a1e62c771d56a9) #x015e8798b1dc755a))
(constraint (= (f #x7500b04e4de1144d) #x01d402c139378451))
(constraint (= (f #xc23e0e2bd2c2a586) #x0308f838af4b0a96))
(constraint (= (f #xe44eb612b10e84a2) #x03913ad84ac43a12))
(constraint (= (f #xaca072aea93e5975) #x0f5e097f3fb42eb9))
(constraint (= (f #x2165631dea166c5e) #x063afa5263e3ab4e))
(constraint (= (f #x5e879762651d822c) #x017a1e5d89947608))
(constraint (= (f #x59a4e931a31ee36c) #x016693a4c68c7b8d))
(constraint (= (f #xceebd2ead39bcdec) #x033baf4bab4e6f37))
(constraint (= (f #x8cced763b72ea5dc) #x0955379a4d973ee6))
(constraint (= (f #x910925ab424be526) #x02442496ad092f94))
(constraint (= (f #xa548e4ce0894dcb5) #x0efd92d5219bd65d))
(constraint (= (f #xad166dd2467ee7ed) #x02b459b74919fb9f))
(constraint (= (f #x78a381c81b091003) #x01e28e07206c2440))
(constraint (= (f #x50eadeb02a7c2135) #x0f13f63d07e84635))
(constraint (= (f #x0be4a3cb7eb8ebd9) #x01c2de45d83c93c6))
(constraint (= (f #x38563ce6e590a44e) #x00e158f39b964291))
(constraint (= (f #xdea97e8e18d817b0) #x063fb8392296838d))
(constraint (= (f #x1ae33d2e31bc3295) #x02f25477252c457b))
(constraint (= (f #xca1b0d607372b50d) #x03286c3581cdcad4))
(constraint (= (f #xe519e589e920debd) #x02f2a2e9a3b6163c))
(constraint (= (f #x4530e1242b4a4075) #x0cf51236c7ddec09))
(constraint (= (f #x47cdcca9c8e8e756) #x0c85655fa593929f))
(constraint (= (f #x12844750013e8ea4) #x004a111d4004fa3a))
(constraint (= (f #xedc9ce28c01e20ce) #x03b72738a3007883))
(constraint (= (f #xdb043b44e85e626b) #x036c10ed13a17989))
(constraint (= (f #xeee837b6de4e2de4) #x03bba0dedb7938b7))
(constraint (= (f #x4aad49c550a45be3) #x012ab5271542916f))
(constraint (= (f #x0969c8a1ecea5bb4) #x01bba59e2353eecd))
(constraint (= (f #xa76100d9d86ed261) #x029d84036761bb49))
(constraint (= (f #x9c0227edeab51ae1) #x0270089fb7aad46b))
(constraint (= (f #xbe368aee6d160211) #x0c25b9f32b73a063))
(constraint (= (f #x066a620287a5a613) #x00abea60788eeea3))
(constraint (= (f #x25e883e966645a95) #x06e39843baaacefb))
(constraint (= (f #x05a984e7496d65cb) #x0016a6139d25b597))
(constraint (= (f #xdc43359283ba98ba) #x064c55eb784cfa9c))
(constraint (= (f #xea30b6de35aedd3a) #x03e51db625ef3674))
(constraint (= (f #x286ada832d64a318) #x078bf6f8577ade52))
(constraint (= (f #x50aa0eece8b28b8e) #x0142a83bb3a2ca2e))
(constraint (= (f #x5ee16ec25a1802e0) #x017b85bb0968600b))
(constraint (= (f #x3e0316a833c8351d) #x042053bf854585f2))
(constraint (= (f #xb3362e484ed4bb4c) #x02ccd8b9213b52ed))
(constraint (= (f #x6e912de1677ee4a2) #x01ba44b7859dfb92))
(constraint (= (f #x5dc3576b74947674) #x0e645f9bd9dbc9a9))
(constraint (= (f #x3c5bc1e98e4d0797) #x044ec423a92d708b))
(constraint (= (f #x1529b84ecece889a) #x03f7ac8d3535399a))
(constraint (= (f #xc616477597d578a0) #x0318591dd65f55e2))
(constraint (= (f #x361eccd04d12aa1e) #x05a235570d737fe2))
(constraint (= (f #xacddeb33b4573906) #x02b377acced15ce4))
(constraint (= (f #x1b5bebd4e461e60d) #x006d6faf53918798))
(constraint (= (f #x2138ea5eb5d76d29) #x0084e3a97ad75db4))
(constraint (= (f #x06aace9201a193ba) #x00bff53b602e2b4c))
(check-synth)
