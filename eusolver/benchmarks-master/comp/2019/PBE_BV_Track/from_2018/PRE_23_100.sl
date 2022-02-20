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

(constraint (= (f #x2d715247669e3eee) #xd28eadb89961c111))
(constraint (= (f #x6454784d46bed069) #x99aa803229000290))
(constraint (= (f #x229e8a5c7ed00ec4) #xcc12307541c7e9d9))
(constraint (= (f #xede7ec3e0a26b15e) #x121813c1f5d94ea1))
(constraint (= (f #xbd2a58bb9e2e5a3d) #x4005024440110040))
(constraint (= (f #x2b03ce3e93389737) #xd44c010004c46088))
(constraint (= (f #xc27d74e85ba7e4d2) #x3d828b17a4581b2d))
(constraint (= (f #x13c15e20295e933e) #xec3ea1dfd6a16cc1))
(constraint (= (f #x6a036e522386db9a) #x95fc91addc792465))
(constraint (= (f #x2e0969464deb7476) #xd1f696b9b2148b89))
(constraint (= (f #xe21d65da588865d5) #x11c2082002771822))
(constraint (= (f #x27e2b13b17beae7e) #xd81d4ec4e8415181))
(constraint (= (f #x757e35622ab1c8cb) #x88800889d5442330))
(constraint (= (f #x3d4da430e92cdd69) #xc022018c10412200))
(constraint (= (f #x64aae2dbed86ee62) #x9b551d241279119d))
(constraint (= (f #xbe7c6127c4c5cea1) #x400018c803322114))
(constraint (= (f #x2d17229a9e433bb5) #xd0288d444018c440))
(constraint (= (f #xca6528a2e4db6074) #xd068430ba8b6ef51))
(constraint (= (f #xce6e6236c11c40e6) #x31919dc93ee3bf19))
(constraint (= (f #xd6ad5777ec578c31) #x201028880128030c))
(constraint (= (f #x780abadd85898011) #x80754402222667ee))
(constraint (= (f #xa796115c722169ed) #x50008ea208dc8000))
(constraint (= (f #x90cb94ee01735e13) #x663042111e88800c))
(constraint (= (f #xc205e1aa477b6e1a) #x3dfa1e55b88491e5))
(constraint (= (f #x1189084e0d64b9ee) #xee76f7b1f29b4611))
(constraint (= (f #xbe75c1ad177d083e) #x418a3e52e882f7c1))
(constraint (= (f #x03a196655b190188) #xfa8d9e67f75a7db3))
(constraint (= (f #xb0832b9bc5dd2ae0) #xf73b3e9657343faf))
(constraint (= (f #x59d7e73eebbc89ee) #xa62818c114437611))
(constraint (= (f #xeece707a779a5cae) #x11318f858865a351))
(constraint (= (f #x5b76c54caed16939) #xa00812a311028044))
(constraint (= (f #xeb92708cede5e807) #x1044887310000178))
(constraint (= (f #x70ee98192be64848) #x569a1bda3e269393))
(constraint (= (f #xe8c5e960510375c3) #x11320009aaec8820))
(constraint (= (f #x65aa02b37adec60d) #x98055d4480001192))
(constraint (= (f #x14ca51867a80eba9) #xea310a6180571044))
(constraint (= (f #xa4468dc8e644b5d7) #x51b91223119b0020))
(constraint (= (f #x18e359d9e77011ee) #xe71ca626188fee11))
(constraint (= (f #xcd08c3e1e9c57591) #x3227300000228826))
(constraint (= (f #x89614bb32c95dd33) #x7608a044c122220c))
(constraint (= (f #x53b8423b68ba02e7) #xa84439c401445d10))
(constraint (= (f #xae9b3e53a18372a6) #x5164c1ac5e7c8d59))
(constraint (= (f #x696c751c1dd4eae9) #x900108a222221110))
(constraint (= (f #x5685eee131499384) #x7e3719ae3611a2b9))
(constraint (= (f #x62cd08a45bea25a7) #x99122751a0015800))
(constraint (= (f #x4266ea2da5ae0eb7) #xb999115000011100))
(constraint (= (f #x91960e59b150574d) #x6660910244aaa882))
(constraint (= (f #x37bc4b6e40e9eb43) #xc80030011b100008))
(constraint (= (f #x743a35b4ae66e663) #x8884480011191198))
(constraint (= (f #xc360a4228be54328) #xdaef09cc2e281b43))
(constraint (= (f #xc4c57e4685a392a5) #x3332801912044450))
(constraint (= (f #xacbb6ea104e6ea6e) #x5344915efb191591))
(constraint (= (f #xd80c824d1ceaca6e) #x27f37db2e3153591))
(constraint (= (f #xb377eaa7775c2ba1) #x4488015088821444))
(constraint (= (f #x45949ea1d5ea2228) #x97a1120d3f20ccc3))
(constraint (= (f #x39ee7a2a6693aec3) #xc401005519044110))
(constraint (= (f #x46c0ab6e2210c351) #xb91354011dce308a))
(constraint (= (f #x42440e4917ed2bb7) #xb99bb11268000440))
(constraint (= (f #xd3e587d0a2ee87e2) #x2c1a782f5d11781d))
(constraint (= (f #x055c28854edb71a3) #xfaa21572a1000844))
(constraint (= (f #x1ded025ae5d3a14c) #xd31c7c77a7428e0d))
(constraint (= (f #x0a88de72bb35a00e) #xf577218d44ca5ff1))
(constraint (= (f #x42c6bbe1809626e8) #x9bd5e62dbf1ec5a3))
(constraint (= (f #x395717b58701b3e9) #xc4288800208e4400))
(constraint (= (f #xb60670eecd43921e) #x49f98f1132bc6de1))
(constraint (= (f #xbdedd6d63dd9a193) #x4000200080224464))
(constraint (= (f #x5ee5ab039d836337) #xa010044c422488c8))
(constraint (= (f #x8eebc28b27b62c89) #x7110015448009136))
(constraint (= (f #x1e17d5e72912eeed) #xe0080200846c1110))
(constraint (= (f #x5852aa94733eeb1d) #xa228554288c01042))
(constraint (= (f #x341500e60dee0ae6) #xcbeaff19f211f519))
(constraint (= (f #x96d3ed5227ca7c5a) #x692c12add83583a5))
(constraint (= (f #xd7b0e48623e057b6) #x284f1b79dc1fa849))
(constraint (= (f #x60409876b62769b5) #x99bb660800988040))
(constraint (= (f #xd9e4766a3a379e97) #x2201889144480000))
(constraint (= (f #x68a3288892a552ac) #x630b4333240803fd))
(constraint (= (f #x5156aee15de43ccd) #xaaa81110a2018032))
(constraint (= (f #x9a31ce8ae7e1ded7) #x644c211510002000))
(constraint (= (f #x3992c675cc75e0b9) #xc464118823080144))
(constraint (= (f #x7dd4c23e102ca4b5) #x802231c00ed11100))
(constraint (= (f #xd8c37c549598c825) #x2230802a22263358))
(constraint (= (f #x8deb4d473e2eddb2) #x7214b2b8c1d1224d))
(constraint (= (f #x70d82ea009341324) #x56bbba0ff231e349))
(constraint (= (f #xa4724e58dc3b60ec) #x09548a7ab5a6ee9d))
(constraint (= (f #x65e14d7a26ce82ea) #x9a1eb285d9317d15))
(constraint (= (f #x05785058ade8a707) #xfa802aa250015088))
(constraint (= (f #x64d1d096e0a071ea) #x9b2e2f691f5f8e15))
(constraint (= (f #xa0be46ec5e5778e1) #x5540191120088010))
(constraint (= (f #x72c27e8496d30dee) #x8d3d817b692cf211))
(constraint (= (f #x250b7e13e5d4dd5d) #xd8a4000c00222222))
(constraint (= (f #xe0243ac887256b09) #x11d9841370888046))
(constraint (= (f #xe771e7a5303bd1ed) #x108800008cc40200))
(constraint (= (f #x81e4e94001420614) #x3d28a21ffe1cf6e1))
(constraint (= (f #x715c7eb1e790ce19) #x88a2000400063106))
(constraint (= (f #x12674ed1d55ec9bd) #xec98810222a01240))
(constraint (= (f #x58c537e5d1e63a7e) #xa73ac81a2e19c581))
(constraint (= (f #xb6c44d080347677e) #x493bb2f7fcb89881))
(constraint (= (f #x9bd024bc6ecd28e6) #x642fdb439132d719))
(constraint (= (f #xdc7cb8e76be59301) #x22000410800024ce))
(constraint (= (f #x5403ea7ac34dda57) #xaabc010010822008))

(check-synth)

