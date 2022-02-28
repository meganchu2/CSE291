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

(constraint (= (f #x979dc93d9b8054d1) #x034311b61323fd59))
(constraint (= (f #x90c20cbde0186c42) #x0379ef9a10ff3c9d))
(constraint (= (f #x37bd4a19ce4c9973) #x064215af318d9b34))
(constraint (= (f #x7bd0b27e5a5c7ea0) #x04217a6c0d2d1c0a))
(constraint (= (f #xc24a16e4716cde9c) #x01edaf48dc74990b))
(constraint (= (f #xe6aa92b69ee55e90) #x0000cd55256d3dca))
(constraint (= (f #xc9d7b3ee5334790b) #x01b142608d665c37))
(constraint (= (f #xced8521011068e4a) #x01893d6f7f77cb8d))
(constraint (= (f #x66b8eb72158568e2) #x0000cd71d6e42b0a))
(constraint (= (f #x223e519a2eae35ec) #x06ee0d732e8a8e50))
(constraint (= (f #xc8b94c8e5cab5e12) #x00009172991cb956))
(constraint (= (f #xab589a0ee814c632) #x02a53b2f88bf59ce))
(constraint (= (f #xa521aa00c23bd570) #x00004a4354018477))
(constraint (= (f #x46bb79c81545e5e1) #x00008d76f3902a8b))
(constraint (= (f #xc0e455ec12aa97c4) #x01f8dd509f6aab41))
(constraint (= (f #x8a5e45e407ea6339) #x000014bc8bc80fd4))
(constraint (= (f #x92a10d740bc2322b) #x000025421ae81784))
(constraint (= (f #x27a4e194eb92d2d0) #x06c2d8f358a36969))
(constraint (= (f #x859cb265eb475c2e) #x00000b3964cbd68e))
(constraint (= (f #x51aa1852d9c8e97c) #x0572af3d6931b8b4))
(constraint (= (f #x173990b065ee5884) #x0746337a7cd08d3b))
(constraint (= (f #xe40dce1c4ba25e8e) #x00df918f1da2ed0b))
(constraint (= (f #xe6ebb8be29a02435) #x00c8a23a0eb2fede))
(constraint (= (f #x56b765365bc46ebb) #x054a44d64d21dc8a))
(constraint (= (f #x8dc93a5d11d45ab6) #x0391b62d17715d2a))
(constraint (= (f #xe6e269ee0a672454) #x0000cdc4d3dc14ce))
(constraint (= (f #x6d5a5e9aeb6d16c3) #x0000dab4bd35d6da))
(constraint (= (f #x824835a76ce233c6) #x03edbe52c498ee61))
(constraint (= (f #xa30e5ace20ab5cc3) #x02e78d298efaa519))
(constraint (= (f #x5042cc3b009dd310) #x0000a0859876013b))
(constraint (= (f #xe4d63731eab1ec30) #x0000c9ac6e63d563))
(constraint (= (f #x51c207d67537262e) #x0000a3840facea6e))
(constraint (= (f #xbbeec2a2539c45a0) #x022089eaed631dd2))
(constraint (= (f #x125a8801028b2e51) #x076d2bbff7eba68d))
(constraint (= (f #x87ecd3dd4e94c303) #x03c09961158b59e7))
(constraint (= (f #x7507e2cb1ee0a898) #x0457c0e9a708fabb))
(constraint (= (f #xb8559502b353c113) #x023d5357ea6561f7))
(constraint (= (f #x4bac12e72588eeec) #x05a29f68c6d3b888))
(constraint (= (f #x697b6860b43e5c5c) #x04b424bcfa5e0d1d))
(constraint (= (f #x030dd84e0debe17d) #x07e7913d8f90a0f4))
(constraint (= (f #x598811ee0355c24e) #x0000b31023dc06ab))
(constraint (= (f #xbda50e20d7eb7b81) #x0212d78ef940a423))
(constraint (= (f #x79daac638e074621) #x04312a9ce38fc5ce))
(constraint (= (f #xd4b91e5dc8aaeb73) #x0000a9723cbb9155))
(constraint (= (f #xd9ada6372779209b) #x0000b35b4c6e4ef2))
(constraint (= (f #x195a14e671b6e628) #x07352f58cc7248ce))
(constraint (= (f #xc90706b792473825) #x01b7c7ca436dc63e))
(constraint (= (f #x6608e9de64372596) #x0000cc11d3bcc86e))
(constraint (= (f #x3d7ee12603dde41b) #x00007afdc24c07bb))
(constraint (= (f #x03b15a4acd6edce1) #x00000762b4959add))
(constraint (= (f #x00b57a70d6eb5bb7) #x07fa542c7948a522))
(constraint (= (f #xcea6c5e5a61e85ec) #x018ac9d0d2cf0bd0))
(constraint (= (f #x32c9538d741ea103) #x00006592a71ae83d))
(constraint (= (f #x594a6ea7a129b0e8) #x0000b294dd4f4253))
(constraint (= (f #x1b13c60b6cb00eee) #x072761cfa49a7f88))
(constraint (= (f #x2ce2ded9ed6eda79) #x000059c5bdb3dadd))
(constraint (= (f #x9d1c1ee8b9c981c2) #x00003a383dd17393))
(constraint (= (f #x4dec8bd4c8cc76ab) #x05909ba159b99c4a))
(constraint (= (f #xc78674d142b9bd2e) #x00008f0ce9a28573))
(constraint (= (f #x77a7be4587441dd3) #x0442c20dd3c5df11))
(constraint (= (f #x8879aebeec0b3a1a) #x000010f35d7dd816))
(constraint (= (f #xad2ea5e362662dea) #x02968ad0e4ecce90))
(constraint (= (f #x3dc90ab84d62d1be) #x0611b7aa3d94e972))
(constraint (= (f #x64beda30467012ee) #x04da092e7dcc7f68))
(constraint (= (f #x8a32531907a02ec4) #x03ae6d6737c2fe89))
(constraint (= (f #xe9b4ae7759ba5be5) #x0000d3695ceeb374))
(constraint (= (f #xab744ebea4e4da73) #x02a45d8a0ad8d92c))
(constraint (= (f #x3bd660c84732dbbc) #x06214cf9bdc66922))
(constraint (= (f #xa9ed54b79754143d) #x02b0955a43455f5e))
(constraint (= (f #x473544161d3551e1) #x00008e6a882c3a6a))
(constraint (= (f #x1604a7e27e515821) #x00002c094fc4fca2))
(constraint (= (f #x5106cdaedede3675) #x0000a20d9b5dbdbc))
(constraint (= (f #x401e2b9b2a83e098) #x0000803c57365507))
(constraint (= (f #xb3d4c4e3d19137b0) #x000067a989c7a322))
(constraint (= (f #x28c89e21ee3a3491) #x000051913c43dc74))
(constraint (= (f #xe9abe15ee3eb5656) #x0000d357c2bdc7d6))
(constraint (= (f #x1783495240abbe25) #x0743e5b56dfaa20e))
(constraint (= (f #x6e26e90ec595a416) #x0000dc4dd21d8b2b))
(constraint (= (f #x896345ec894481d1) #x03b4e5d09bb5dbf1))
(constraint (= (f #xebdeba06e7e94e15) #x0000d7bd740dcfd2))
(constraint (= (f #xe6c35421e171d642) #x0000cd86a843c2e3))
(constraint (= (f #x82281b656e3135eb) #x0000045036cadc62))
(constraint (= (f #xabde03eeec0ad778) #x02a10fe0889fa944))
(constraint (= (f #x73711c6ec1257bc6) #x0000e6e238dd824a))
(constraint (= (f #x26e37809b8eece37) #x00004dc6f01371dd))
(constraint (= (f #x908d901eea5a6d67) #x0000211b203dd4b4))
(constraint (= (f #x9322e598e2be6e21) #x00002645cb31c57c))
(constraint (= (f #xbc04174c491c8a36) #x021fdf459db71bae))
(constraint (= (f #xc5ec42a8cceceb1d) #x01d09deab99898a7))
(constraint (= (f #x4b5a53b3dd74e91b) #x05a52d62611458b7))
(constraint (= (f #x1e921daebbe03abc) #x070b6f128a20fe2a))
(constraint (= (f #x92e00ec160544921) #x0368ff89f4fd5db6))
(constraint (= (f #x4d554a8ed37a620d) #x00009aaa951da6f4))
(constraint (= (f #x4edb7c5bc254e87d) #x0589241d21ed58bc))
(constraint (= (f #x6a3196db025d108a) #x0000d4632db604ba))
(constraint (= (f #xb6cba5a83ec762e8) #x00006d974b507d8e))
(constraint (= (f #xa32e154d3cbe8d54) #x02e68f55961a0b95))
(constraint (= (f #x42be5832c58947c6) #x0000857cb0658b12))
(constraint (= (f #x255b4d57a75e2b6e) #x06d5259542c50ea4))
(constraint (= (f #x534d959ceed31c83) #x056593531889671b))

(check-synth)
