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

(constraint (= (f #xc719923beae19a2a) #x96d01630a89f0710))
(constraint (= (f #xb2aa73ee5c50b009) #x6554e7dcb8a16012))
(constraint (= (f #x71b8ce038ed5d9ed) #xe3719c071dabb3da))
(constraint (= (f #x78091d0eece693dc) #xff1319bc0451f5c2))
(constraint (= (f #xa2d1e426ae6275cb) #x45a3c84d5cc4eb96))
(constraint (= (f #xcb1a962e37e71e71) #x96352c5c6fce3ce2))
(constraint (= (f #x11beee09a6c6bb44) #x214a01d27955a1e0))
(constraint (= (f #x70e0467b0e0909ce) #xefdc84397dd332a4))
(constraint (= (f #xd76dc93cd0d53e36) #xb4362b5e3bb0dbaa))
(constraint (= (f #xe1a33346ce5e5673) #xc346668d9cbcace6))
(constraint (= (f #x3c1c5414c925272b) #x7838a829924a4e56))
(constraint (= (f #x8e5eb7ce960e2eed) #x1cbd6f9d2c1c5dda))
(constraint (= (f #xeeca94e5ceeb9bc2) #xc04c7b57240a44fc))
(constraint (= (f #xbeaa474becead0c5) #x7d548e97d9d5a18a))
(constraint (= (f #x874abb4dd9d44747) #x0e95769bb3a88e8e))
(constraint (= (f #xa01a19d86ee5598b) #x403433b0ddcab316))
(constraint (= (f #xce62aa13cee289c1) #x9cc554279dc51382))
(constraint (= (f #x59c62a706ea91b0d) #xb38c54e0dd52361a))
(constraint (= (f #xb49345e9039be602) #x7fb4e36f2744b0c4))
(constraint (= (f #x26b8867eda8d416c) #x49a61c326e4b2af4))
(constraint (= (f #xc0d657dd5b2ad991) #x81acafbab655b322))
(constraint (= (f #x1a58e939b74d875c) #x37facf545872be52))
(constraint (= (f #x2aee7d7e5e2a4259) #x55dcfafcbc5484b2))
(constraint (= (f #xe8450aa5b4c0e1b9) #xd08a154b6981c372))
(constraint (= (f #x4ee36e652c585ee3) #x9dc6dcca58b0bdc6))
(constraint (= (f #xe998d7e3adda86ed) #xd331afc75bb50dda))
(constraint (= (f #xba9bcaab5e191b73) #x75379556bc3236e6))
(constraint (= (f #x8954a3e89c45e373) #x12a947d1388bc6e6))
(constraint (= (f #xcee02a28a3dc694d) #x9dc0545147b8d29a))
(constraint (= (f #x2dd645b6e6d8c16a) #x5e1643db116a9af8))
(constraint (= (f #x0b857d34ed4384e3) #x170afa69da8709c6))
(constraint (= (f #x642572d690c9d8a1) #xc84ae5ad2193b142))
(constraint (= (f #xbae1ece046b0eb09) #x75c3d9c08d61d612))
(constraint (= (f #xced17e1b19ace1dc) #x8478d3f5506c5f82))
(constraint (= (f #x2934c68ee5ce7404) #x574f15cc17252688))
(constraint (= (f #x7cebee51d3a07be4) #xf64aa1699d34f8b4))
(constraint (= (f #xb1387c23eca7a642) #x7457f7c3a4dbb84c))
(constraint (= (f #x94daa0c82a5054eb) #x29b5419054a0a9d6))
(constraint (= (f #x2bd6d5c0522b294c) #x52d77138ae1337b0))
(constraint (= (f #xd17e51adae80ea91) #xa2fca35b5d01d522))
(constraint (= (f #x099124edd246d8b6) #x12106d461ec56a7a))
(constraint (= (f #xed76d2c28ecdd9e5) #xdaeda5851d9bb3ca))
(constraint (= (f #x47386be5ebedb6e1) #x8e70d7cbd7db6dc2))
(constraint (= (f #xe7de4b81949373c2) #xd3475e731bb489fc))
(constraint (= (f #xd0d7cdda670ae9d4) #xbbb5620f82f48e92))
(constraint (= (f #x6e2d2078a9c28015) #xdc5a40f15385002a))
(constraint (= (f #x5825507e53eeae25) #xb04aa0fca7dd5c4a))
(constraint (= (f #xe326ee2a5e0a2ead) #xc64ddc54bc145d5a))
(constraint (= (f #x34635581c48270aa) #x6e4ac1b3b194af40))
(constraint (= (f #xe2801a6c8e97ed03) #xc50034d91d2fda06))
(constraint (= (f #xa3eed89788926bdb) #x47ddb12f1124d7b6))
(constraint (= (f #x92b5922ac8207073) #x256b24559040e0e6))
(constraint (= (f #xe4bbe488b930c258) #xd5e0b58065479cfa))
(constraint (= (f #x7ce651967e9ca530) #xf650691e32eadec6))
(constraint (= (f #xda9401541a68e6be) #xae7a8282b79cd1aa))
(constraint (= (f #xbaeba8365e5e1058) #x628a256a7777e2ba))
(constraint (= (f #x8c6307817b5dabee) #x094a6ff2d9d0e2a0))
(constraint (= (f #x114d7076de5addde) #x20b34ee3677ee006))
(constraint (= (f #x09573b4c7b8ec893) #x12ae7698f71d9126))
(constraint (= (f #x19500525d2d19ebb) #x32a00a4ba5a33d76))
(constraint (= (f #xb6eea0c5ad9531de) #x7b009593ee98c586))
(constraint (= (f #xb9c4d49a056106ee) #x64b133a74a6e2d00))
(constraint (= (f #xda8b186ec35e125d) #xb51630dd86bc24ba))
(constraint (= (f #xd371970652ce010b) #xa6e32e0ca59c0216))
(constraint (= (f #x1e0d5e26085e26d5) #x3c1abc4c10bc4daa))
(constraint (= (f #x0a8b44727eae9a63) #x151688e4fd5d34c6))
(constraint (= (f #x5e6c0577918ca634) #xb7158a41d128d8ae))
(constraint (= (f #xe9301864d2aacc4d) #xd26030c9a555989a))
(constraint (= (f #xec395527c794dd16) #xc5f580eb77db218e))
(constraint (= (f #xe14be48b5e3e0aaa) #xdebeb587d7bbd400))
(constraint (= (f #x6225da54aaee4134) #xc80f0fe3c0814a4e))
(constraint (= (f #xeab1609591a6eb55) #xd562c12b234dd6aa))
(constraint (= (f #x7bdacae93589ec7d) #xf7b595d26b13d8fa))
(constraint (= (f #xedbea8ed7a49e13a) #xc6ca84c75bdafe52))
(constraint (= (f #xd64990219d600a43) #xac9320433ac01486))
(constraint (= (f #x2c32473623509eee) #x5de2c68a82cb2e00))
(constraint (= (f #xcbbeb4e879c56990) #x8e0abf4dfcb27e12))
(constraint (= (f #x3ce6c6bc8a95e112) #x7e5155ae84797e06))
(constraint (= (f #x70e85e784e1279db) #xe1d0bcf09c24f3b6))
(constraint (= (f #x187bbe48b88d8123) #x30f77c91711b0246))
(constraint (= (f #x26ad546e3e9cee4c) #x498f0251baea4150))
(constraint (= (f #x37699ea4e7ed124a) #x683e0e9d532786dc))
(constraint (= (f #x1d96ee5b53305d1d) #x3b2ddcb6a660ba3a))
(constraint (= (f #xdbed4394d14ad1b0) #xaca72f5b38bcf956))
(constraint (= (f #x3d99e2e5135da041) #x7b33c5ca26bb4082))
(constraint (= (f #x0ea68ba3ecbac6ca) #x1c99c633a4e2d54c))
(constraint (= (f #x4128ded5ebd7762d) #x8251bdabd7aeec5a))
(constraint (= (f #xbb22888993a2e0b0) #x6121400215319d76))
(constraint (= (f #x54ae5ab2582aa73e) #xa3c97e32fb501a9a))
(constraint (= (f #x4199e34bead56cc0) #x8b00fafea8f07418))
(constraint (= (f #x88e6c31e138a27a9) #x11cd863c27144f52))
(constraint (= (f #x5010398ee8417a5a) #xaa22742c0d8adbfe))
(constraint (= (f #xbeede093153b0d22) #x6a067d3448d17be0))
(constraint (= (f #xa72e079e55a7d3ed) #x4e5c0f3cab4fa7da))
(constraint (= (f #x9e021ed1adde72d1) #x3c043da35bbce5a2))
(constraint (= (f #xd562e92e9e00d656) #xb0698f78efc1b666))
(constraint (= (f #x321680eb8e7b3039) #x642d01d71cf66072))
(constraint (= (f #x2869774552b67c96) #x55dfc0620f3a36be))
(constraint (= (f #xb5abe12ad91b2508) #x7de2be70e9152eb0))
(constraint (= (f #x01ee3c5c802ee5c1) #x03dc78b9005dcb82))

(check-synth)

