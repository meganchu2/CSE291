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

(constraint (= (f #x0c21e84200506ad2) #x00000c21e8420050))
(constraint (= (f #xc44826220eddbe69) #x0000c44826220edd))
(constraint (= (f #xd17c504d4ae7ece2) #x0000d17c504d4ae7))
(constraint (= (f #xa0e2b167bae1402d) #x0000a0e2b167bae1))
(constraint (= (f #xee837e3e3ee6ecdc) #x0000ee837e3e3ee6))
(constraint (= (f #x5ae90b7e5b96cd15) #x00005ae90b7e5b96))
(constraint (= (f #x8ee3642716e3ae4c) #x00008ee3642716e3))
(constraint (= (f #x3d39018a05a61782) #x00003d39018a05a6))
(constraint (= (f #xa647a29e4b1bee7e) #x0000a647a29e4b1b))
(constraint (= (f #x99789ee5d51ed5ec) #x000099789ee5d51e))
(constraint (= (f #x225071e26375ed7b) #x0000225071e26375))
(constraint (= (f #xa849c692ba69e68e) #x0000a849c692ba69))
(constraint (= (f #x7c3702295e8de0da) #x00007c3702295e8d))
(constraint (= (f #x6bc7bea9a6ae0111) #x00006bc7bea9a6ae))
(constraint (= (f #x6c7b97edc6ed073e) #x00006c7b97edc6ed))
(constraint (= (f #x7ca82e84ed991a02) #x00007ca82e84ed99))
(constraint (= (f #x1e610e9844d5ae6d) #x00001e610e9844d5))
(constraint (= (f #xba640c06a350101b) #x0000ba640c06a350))
(constraint (= (f #x5de4b68008a5e6de) #x00005de4b68008a5))
(constraint (= (f #xeebb9ee080e3e1e8) #x0000eebb9ee080e3))
(constraint (= (f #x1a1e7e3c4b5b9d6e) #x00001a1e7e3c4b5b))
(constraint (= (f #x7502e733daced491) #x00007502e733dace))
(constraint (= (f #xeeb14357014e7e0c) #x0000eeb14357014e))
(constraint (= (f #xeebbe3e9d3b72e9e) #x0000eebbe3e9d3b7))
(constraint (= (f #xc0689eb2412cb087) #x0000c0689eb2412c))
(constraint (= (f #xd76762ec29ac847e) #x0000d76762ec29ac))
(constraint (= (f #x54c07c05c1ae50b8) #x000054c07c05c1ae))
(constraint (= (f #x2b17c02dc52843de) #x00002b17c02dc528))
(constraint (= (f #x73b8060a933187e2) #x000073b8060a9331))
(constraint (= (f #x9c7d9d19a791e259) #x00009c7d9d19a791))
(constraint (= (f #x9854425e7ec728b8) #x00009854425e7ec7))
(constraint (= (f #x8114eb4101ed1b7e) #x00008114eb4101ed))
(constraint (= (f #xddb38e0d2399cede) #x0000ddb38e0d2399))
(constraint (= (f #x2b74b0912de9174a) #x00002b74b0912de9))
(constraint (= (f #xc8ebe7d9c3d150d6) #x0000c8ebe7d9c3d1))
(constraint (= (f #x5b6a263ec7e1de7a) #x00005b6a263ec7e1))
(constraint (= (f #x89a8e5846ec2c766) #x000089a8e5846ec2))
(constraint (= (f #xa4bd78e438bd572e) #x0000a4bd78e438bd))
(constraint (= (f #x27e76d4318b33622) #x000027e76d4318b3))
(constraint (= (f #x72e142ee734a3dd8) #x000072e142ee734a))
(constraint (= (f #xbd1845720d1ad840) #x0000bd1845720d1a))
(constraint (= (f #xb78810272371dad3) #x0000b78810272371))
(constraint (= (f #xa40746924dc35e01) #x0000a40746924dc3))
(constraint (= (f #x66a9012013ed8454) #x000066a9012013ed))
(constraint (= (f #xaed775ed6333124d) #x0000aed775ed6333))
(constraint (= (f #xe5e382c995e27e1b) #x0000e5e382c995e2))
(constraint (= (f #x66903cd26c98260d) #x000066903cd26c98))
(constraint (= (f #x517d1339be4e196c) #x0000517d1339be4e))
(constraint (= (f #x7ec04565b277e0e0) #x00007ec04565b277))
(constraint (= (f #xc5da7e8be6b131e9) #x0000c5da7e8be6b1))
(constraint (= (f #x557d18e324e9e6ec) #x0000557d18e324e9))
(constraint (= (f #xd339c95b77eea707) #x0000d339c95b77ee))
(constraint (= (f #xe3eb7d82b3345b18) #x0000e3eb7d82b334))
(constraint (= (f #x6bdd1b3edbec9acb) #x00006bdd1b3edbec))
(constraint (= (f #x659d6dbbc5871a2e) #x0000659d6dbbc587))
(constraint (= (f #xbc2ebec2be84d21d) #x0000bc2ebec2be84))
(constraint (= (f #xe57e3d89e7d42672) #x0000e57e3d89e7d4))
(constraint (= (f #x63d9aedede17dae8) #x000063d9aedede17))
(constraint (= (f #xcedcae7e211a5a81) #x0000cedcae7e211a))
(constraint (= (f #x2e877ab3a9101aa8) #x00002e877ab3a910))
(constraint (= (f #x6c7660516e163110) #x00006c7660516e16))
(constraint (= (f #x5450be76ce1edb4e) #x00005450be76ce1e))
(constraint (= (f #x2c4e6c450132944b) #x00002c4e6c450132))
(constraint (= (f #x1c45a21c33b0784d) #x00001c45a21c33b0))
(constraint (= (f #xd75e75e27a519e61) #x0000d75e75e27a51))
(constraint (= (f #x4880e90825770107) #x00004880e9082577))
(constraint (= (f #x6b440dbe0887b24a) #x00006b440dbe0887))
(constraint (= (f #x9e935eb789d6287e) #x00009e935eb789d6))
(constraint (= (f #x4eae1c3aeedee313) #x00004eae1c3aeede))
(constraint (= (f #xc492a2c6e879ecde) #x0000c492a2c6e879))
(constraint (= (f #x74a42b6a03336490) #x000074a42b6a0333))
(constraint (= (f #xea86e91e0c36ee2d) #x0000ea86e91e0c36))
(constraint (= (f #x0725c0c10b0e3a2a) #x00000725c0c10b0e))
(constraint (= (f #x338b4b2e49c65bc4) #x0000338b4b2e49c6))
(constraint (= (f #xe393e94906e82e63) #x0000e393e94906e8))
(constraint (= (f #xe38ba3b315bb3388) #x0000e38ba3b315bb))
(constraint (= (f #xac10955ded097044) #x0000ac10955ded09))
(constraint (= (f #xe795374c59b7468d) #x0000e795374c59b7))
(constraint (= (f #x91824a037cdc676d) #x000091824a037cdc))
(constraint (= (f #xa33c4e58d17b98e0) #x0000a33c4e58d17b))
(constraint (= (f #x66c3c5b0aad79035) #x000066c3c5b0aad7))
(constraint (= (f #x2b54460ca9ec4c74) #x00002b54460ca9ec))
(constraint (= (f #x9e3ab46981339770) #x00009e3ab4698133))
(constraint (= (f #x8b005e13c598e2eb) #x00008b005e13c598))
(constraint (= (f #x7e9cba7e1e05e060) #x00007e9cba7e1e05))
(constraint (= (f #x4ec9152e2e9e7cde) #x00004ec9152e2e9e))
(constraint (= (f #xba365ae119418730) #x0000ba365ae11941))
(constraint (= (f #x33aa2ed5826ee2d9) #x000033aa2ed5826e))
(constraint (= (f #x9572e672724b041a) #x00009572e672724b))
(constraint (= (f #x655a720de0758ce2) #x0000655a720de075))
(constraint (= (f #xb37e5b4230da1895) #x0000b37e5b4230da))
(constraint (= (f #x43abe8aeb2de6639) #x000043abe8aeb2de))
(constraint (= (f #xa5c9ceee590823b5) #x0000a5c9ceee5908))
(constraint (= (f #xe857d4ee4368a05c) #x0000e857d4ee4368))
(constraint (= (f #xbd8be6531e14410c) #x0000bd8be6531e14))
(constraint (= (f #x1d4e0ee644874e32) #x00001d4e0ee64487))
(constraint (= (f #x06eee82ebb773dce) #x000006eee82ebb77))
(constraint (= (f #x766e9beee71e91e7) #x0000766e9beee71e))
(constraint (= (f #xb330bd2bad97d21d) #x0000b330bd2bad97))
(constraint (= (f #x90dca4402d7845e9) #x000090dca4402d78))

(check-synth)

