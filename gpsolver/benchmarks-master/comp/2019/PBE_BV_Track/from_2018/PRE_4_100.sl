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

(constraint (= (f #x1baee10e17dd57e7) #x1baee10e17dd57e7))
(constraint (= (f #xa88b41e7ce2ed738) #xfffffffffffffffe))
(constraint (= (f #x0b2c8e902d4546ce) #xfffffffffffffffe))
(constraint (= (f #xc94e2b482806b295) #xc94e2b482806b295))
(constraint (= (f #xe567399a08120a7e) #xfffffffffffffffe))
(constraint (= (f #xe673c54a9c6eee3a) #xfffffffffffffffe))
(constraint (= (f #x207d32218ea89650) #xfffffffffffffffe))
(constraint (= (f #x14da3336ba0dd831) #x14da3336ba0dd831))
(constraint (= (f #x6e76ba3b218731ac) #xfffffffffffffffe))
(constraint (= (f #xa4a10eebeaaa44ce) #xfffffffffffffffe))
(constraint (= (f #x22110ed11e783d71) #x22110ed11e783d71))
(constraint (= (f #x96b8325d85675c16) #xfffffffffffffffe))
(constraint (= (f #xa0c3aa13c6041168) #xfffffffffffffffe))
(constraint (= (f #x057815e813e02675) #x057815e813e02675))
(constraint (= (f #x26c315785ba40857) #x26c315785ba40857))
(constraint (= (f #xa3a3b759d8cbd628) #xfffffffffffffffe))
(constraint (= (f #x6a72da69ce9687b4) #xfffffffffffffffe))
(constraint (= (f #xecec70d8e0e8e77e) #xfffffffffffffffe))
(constraint (= (f #xa3ccdbce547d2984) #xfffffffffffffffe))
(constraint (= (f #x719e1932da719aac) #xfffffffffffffffe))
(constraint (= (f #xce90138052deae39) #xce90138052deae39))
(constraint (= (f #xd59515e320824515) #xd59515e320824515))
(constraint (= (f #x81a324de4ea03159) #x81a324de4ea03159))
(constraint (= (f #xa4ec7013e15a2c2c) #xfffffffffffffffe))
(constraint (= (f #x46c118e53327eed5) #x46c118e53327eed5))
(constraint (= (f #xc3abc191caae91e5) #xc3abc191caae91e5))
(constraint (= (f #xa5a0ee46a9008119) #xa5a0ee46a9008119))
(constraint (= (f #x1bd5b8bda581a141) #x1bd5b8bda581a141))
(constraint (= (f #x08400e275636806e) #xfffffffffffffffe))
(constraint (= (f #xe3d6e60814a58dbe) #xfffffffffffffffe))
(constraint (= (f #x5602a9ae8e57d57e) #xfffffffffffffffe))
(constraint (= (f #xe162c8447a59ce8d) #xe162c8447a59ce8d))
(constraint (= (f #x9a86eda1ae6c9e1e) #xfffffffffffffffe))
(constraint (= (f #x8e348221e3569555) #x8e348221e3569555))
(constraint (= (f #xd6805b34cbeb40ce) #xfffffffffffffffe))
(constraint (= (f #x433e8d93ce8b4410) #xfffffffffffffffe))
(constraint (= (f #x14d1979ecac5d0d0) #xfffffffffffffffe))
(constraint (= (f #xb360e92d59642eb6) #xfffffffffffffffe))
(constraint (= (f #x373dd43220e5858e) #xfffffffffffffffe))
(constraint (= (f #xc72abe38ac0e16ac) #xfffffffffffffffe))
(constraint (= (f #x46e682a4ee86b569) #x46e682a4ee86b569))
(constraint (= (f #xb9883e64419b39a2) #xfffffffffffffffe))
(constraint (= (f #x37e6677a73b83e50) #xfffffffffffffffe))
(constraint (= (f #xce7cd9646122771a) #xfffffffffffffffe))
(constraint (= (f #x8c34215d4192e9a6) #xfffffffffffffffe))
(constraint (= (f #x2a3a3d1375315e7e) #xfffffffffffffffe))
(constraint (= (f #x8ac783736147040d) #x8ac783736147040d))
(constraint (= (f #x79c7b5ad3ccb573c) #xfffffffffffffffe))
(constraint (= (f #xbe422081eb0e67b9) #xbe422081eb0e67b9))
(constraint (= (f #x2885bc725cc52070) #xfffffffffffffffe))
(constraint (= (f #x45ae0dd9b801e024) #xfffffffffffffffe))
(constraint (= (f #xe664e66a2ce6a279) #xe664e66a2ce6a279))
(constraint (= (f #x6edebd9a9b0ee189) #x6edebd9a9b0ee189))
(constraint (= (f #x6e27ee04e0db48dd) #x6e27ee04e0db48dd))
(constraint (= (f #xa9e34d68cce1bbb5) #xa9e34d68cce1bbb5))
(constraint (= (f #x03030e25c6eb87d2) #xfffffffffffffffe))
(constraint (= (f #x8113cec418ede888) #xfffffffffffffffe))
(constraint (= (f #x4b13500e1d966bb5) #x4b13500e1d966bb5))
(constraint (= (f #x261d055cad9ec636) #xfffffffffffffffe))
(constraint (= (f #xe3b76081d8536da9) #xe3b76081d8536da9))
(constraint (= (f #xed200c19ab5b64d5) #xed200c19ab5b64d5))
(constraint (= (f #xe1788dec9c138e54) #xfffffffffffffffe))
(constraint (= (f #x3244a734ec45c7db) #x3244a734ec45c7db))
(constraint (= (f #x514adba609ab85de) #xfffffffffffffffe))
(constraint (= (f #x42e0ed7be3016aea) #xfffffffffffffffe))
(constraint (= (f #x79c8c7067942381e) #xfffffffffffffffe))
(constraint (= (f #x5a8cb2844043d1a0) #xfffffffffffffffe))
(constraint (= (f #xd39e7e076c628435) #xd39e7e076c628435))
(constraint (= (f #xcd9db259e58aeb87) #xcd9db259e58aeb87))
(constraint (= (f #x4b48e53248b375ce) #xfffffffffffffffe))
(constraint (= (f #xa78c921aaca4cda2) #xfffffffffffffffe))
(constraint (= (f #x640e563e9d6dac97) #x640e563e9d6dac97))
(constraint (= (f #xd7cc1b16347609d5) #xd7cc1b16347609d5))
(constraint (= (f #x1ae1c918baa3dd66) #xfffffffffffffffe))
(constraint (= (f #xec6ba3d198e5b6dd) #xec6ba3d198e5b6dd))
(constraint (= (f #x9ebee0846d2de990) #xfffffffffffffffe))
(constraint (= (f #x3c8e32e783c32cdd) #x3c8e32e783c32cdd))
(constraint (= (f #x3404560e1a29c97e) #xfffffffffffffffe))
(constraint (= (f #xc83c646e5d33a2eb) #xc83c646e5d33a2eb))
(constraint (= (f #x3a12b1c467c00578) #xfffffffffffffffe))
(constraint (= (f #xe2b1e9eb54e60886) #xfffffffffffffffe))
(constraint (= (f #x9d4701ea41d6ec3c) #xfffffffffffffffe))
(constraint (= (f #x329a5a604e143e44) #xfffffffffffffffe))
(constraint (= (f #x06ee200b992ceed6) #xfffffffffffffffe))
(constraint (= (f #x05a4db32197de1a7) #x05a4db32197de1a7))
(constraint (= (f #xeed75e8821b4ea78) #xfffffffffffffffe))
(constraint (= (f #x0587475d6e350d76) #xfffffffffffffffe))
(constraint (= (f #x7a8b3d5c36e2ba64) #xfffffffffffffffe))
(constraint (= (f #xe4c6de6c51222dae) #xfffffffffffffffe))
(constraint (= (f #xe369b86d005179b2) #xfffffffffffffffe))
(constraint (= (f #x488ea9d35ade62e1) #x488ea9d35ade62e1))
(constraint (= (f #xd1ad2b193ee348be) #xfffffffffffffffe))
(constraint (= (f #xca0bb959dd303e30) #xfffffffffffffffe))
(constraint (= (f #xab13ea339863e794) #xfffffffffffffffe))
(constraint (= (f #x9e0eba7e77aa6a9d) #x9e0eba7e77aa6a9d))
(constraint (= (f #xbba532de91574be2) #xfffffffffffffffe))
(constraint (= (f #x2e110cd48529a3d5) #x2e110cd48529a3d5))
(constraint (= (f #x36d2e1ceeee921c4) #xfffffffffffffffe))
(constraint (= (f #x0ca8b56cb9496865) #x0ca8b56cb9496865))
(constraint (= (f #xe9e0205b4c01ec3e) #xfffffffffffffffe))

(check-synth)

