
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


(constraint (= (f #xa42980b32589b07b) #x0000a429a4bba5bb))
(constraint (= (f #xb7e4a6938ce3744b) #xfc9bfdfef73dcbbf))
(constraint (= (f #x1d1ce6b3cbc46d05) #x00001d1cffbfeff7))
(constraint (= (f #x83d2e88ba336c1a0) #x000083d2ebdbebbf))
(constraint (= (f #xdbb261ee1d8d44d8) #x0000dbb2fbfe7def))
(constraint (= (f #xe658ac4028350932) #xf9bf77bffffeffed))
(constraint (= (f #x16d75228682e3b2e) #xffbaafdffffddcdd))
(constraint (= (f #x648e5eebe5d15b39) #x0000648e7eeffffb))
(constraint (= (f #x4a052654581eee59) #x00004a056e557e5e))
(constraint (= (f #x425234ae7d36c1d8) #x0000425276fe7dbe))
(constraint (= (f #x9e44c66e125319b1) #xf7bbbb99fffeee6e))
(constraint (= (f #x52b2584710bee85b) #x000052b25af758ff))
(constraint (= (f #xe3daeae35692c077) #xfde7555debfffff8))
(constraint (= (f #x97806cd20e59996a) #xfefffb3fffbe66fd))
(constraint (= (f #x1586b71eee7a3cdb) #x00001586b79eff7e))
(constraint (= (f #x4e09da20a8c22627) #x00004e09de29fae2))
(constraint (= (f #xed41c653c6e79be8) #xf3bffbbefb99e657))
(constraint (= (f #x54475be98343dc50) #xfbbbae577fffe3bf))
(constraint (= (f #xc2850e2e8d8e4516) #xffffffdd7777bbef))
(constraint (= (f #x9ee29018727c2a6d) #xf71dfffffddbfddb))
(constraint (= (f #x45ab2837731e7e27) #x000045ab6dbf7b3f))
(constraint (= (f #xe808692320b460de) #x0000e808e92b69b7))
(constraint (= (f #x7aa0caa49cae9b0c) #xfd5ff75ff77576ff))
(constraint (= (f #x3eb2398a9b9ecca9) #xfd5dde7776673377))
(constraint (= (f #xe8c21a170eeae3b8) #x0000e8c2fad71eff))
(constraint (= (f #xe64853d4a9059a9e) #x0000e648f7dcfbd5))
(constraint (= (f #x9b976325e6a22300) #x00009b97fbb7e7a7))
(constraint (= (f #xeaeddee57304322e) #xf553231bacfffddd))
(constraint (= (f #x9ee5533ee8c7ea57) #xf71baecd177b95fa))
(constraint (= (f #x0e53642433121bc1) #x00000e536e777736))
(constraint (= (f #x917e3cc51a7205cc) #xfee9df3befddffb3))
(constraint (= (f #xd242e513d2a0dbce) #xffffdbeeefdff673))
(constraint (= (f #x02ad5e93ebeed674) #xffd7ab7ed5513b9b))
(constraint (= (f #xc436ed5de939eb2e) #xfbfd93aa37ee75dd))
(constraint (= (f #x92bb1421551889b3) #xffd4efffeaef776c))
(constraint (= (f #x4c1e9206e2d12856) #xfbff7fff9dfefffb))
(constraint (= (f #x1d1ec09151aaa649) #xfeef3ffeeef55dbf))
(constraint (= (f #x7b9b1eac1ca86a9b) #x00007b9b7fbf1eac))
(constraint (= (f #x42ab8ee4869919d0) #xffd5771bfff6ee6f))
(constraint (= (f #xaac66d5239932c15) #xf57b9bafde6edffe))
(constraint (= (f #x15cc75a2dc59690e) #xfeb3bafdf3beffff))
(constraint (= (f #x89b0d893d779751d) #x000089b0d9b3dffb))
(constraint (= (f #x5dc6e7c3b5a39ec6) #x00005dc6ffc7f7e3))
(constraint (= (f #x7eaeee25eebc1ba0) #x00007eaefeafeebd))
(constraint (= (f #xe60cebb694000028) #xf9ff354dffffffff))
(constraint (= (f #x405070eec4231514) #xfffffff13bfdeeef))
(constraint (= (f #x5e45840b64ec7167) #x00005e45de4fe4ef))
(constraint (= (f #x45eded76e8990954) #xfbb333a99776ffeb))
(constraint (= (f #xdc27e5ee94446e79) #x0000dc27fdeff5ee))
(constraint (= (f #xdee3398d16ce7a0b) #xf31dce77efb39dff))
(constraint (= (f #xd6e72164753ec574) #xfb99dffbbaed3bab))
(constraint (= (f #x41656e4840b029de) #x000041656f6d6ef8))
(constraint (= (f #x6ee1dc9ea8a24aed) #xf91fe377577dff53))
(constraint (= (f #x1630b433013516b7) #xffdffffcffeeefdc))
(constraint (= (f #x2e4528174bce725b) #x00002e452e576bdf))
(constraint (= (f #x2dc31ce670d14786) #x00002dc33de77cf7))
(constraint (= (f #x1a3ee22ad1919a24) #x00001a3efa3ef3bb))
(constraint (= (f #xc8304e2372007d30) #xf7fffbddcdfffaef))
(constraint (= (f #xc589eeca32117eca) #xfbf77137ddfee937))
(constraint (= (f #x469ee7e110460b30) #xfbf7199feffbffcf))
(constraint (= (f #x390d97d89b29731e) #x0000390dbfdd9ff9))
(constraint (= (f #x212890473d5e9e7a) #x00002128b16fbd5f))
(constraint (= (f #x31259ab7da0ba6ad) #xfeffe75ca7ff5dd7))
(constraint (= (f #xbe86658184b55d4e) #xf57f9bfffffeaabb))
(constraint (= (f #x4e2ae6dbe559aae9) #xfbdd59b65bae7557))
(constraint (= (f #x918665431e7eab2c) #xfeff9bbfef9955df))
(constraint (= (f #xd7a09960ea40397e) #x0000d7a0dfe0fb60))
(constraint (= (f #xe0ecd014be6e94b7) #xfff33ffff5997ffc))
(constraint (= (f #x994eb6858407ab1a) #x0000994ebfcfb687))
(constraint (= (f #x34dea2de5747d8ee) #xffb35df3babba771))
(constraint (= (f #x6e8e4a70beb1029e) #x00006e8e6efefef1))
(constraint (= (f #x49adba6a4a293aed) #xff7765ddffdfed53))
(constraint (= (f #x8798e828ee00e64e) #xffe777ff71fff9bb))
(constraint (= (f #xae60054e78b2ee47) #x0000ae60af6e7dfe))
(constraint (= (f #xbe9ce6913b9c6801) #x0000be9cfe9dff9d))
(constraint (= (f #x3924cc1dc8d58367) #x00003924fd3dccdd))
(constraint (= (f #xe4bb32560cbed20d) #xfbf4cdfbff753fff))
(constraint (= (f #x2bce2347c63d016d) #xfd73ddfbbbdefffb))
(constraint (= (f #xbcca3b6e0e9b8965) #x0000bccabfee3fff))
(constraint (= (f #xca279d535063d3da) #x0000ca27df77dd73))
(constraint (= (f #xd424cad436e0125a) #x0000d424def4fef4))
(constraint (= (f #xa9c31041c74a4c9e) #x0000a9c3b9c3d74b))
(constraint (= (f #x5622c68dec193e55) #xfbddfbf733feedba))
(constraint (= (f #x45a842287bd26623) #x000045a847a87bfa))
(constraint (= (f #xdc9338dee2ee93ca) #xf37ecf731dd17ef7))
(constraint (= (f #xce7cbe854481d9d2) #xf39b757fbbffe66f))
(constraint (= (f #x4deec20bee091abd) #x00004deecfefee0b))
(constraint (= (f #x0ed03069bcdc6e10) #xff3fffff6733b9ff))
(constraint (= (f #x2464769428759e98) #x0000246476f47ef5))
(constraint (= (f #xea271e10eaccc54c) #xf5ddeffff5733bbb))
(constraint (= (f #xc12e535dd7e62065) #x0000c12ed37fd7ff))
(constraint (= (f #xd1ec1ab3689320e9) #xfef3ff5cdf7edff7))
(constraint (= (f #x880d2a16e139be62) #x0000880daa1feb3f))
(constraint (= (f #x7ed42e60a0e6065e) #x00007ed47ef4aee6))
(constraint (= (f #xa8229379883848e9) #xf7fdfece77ffff77))
(constraint (= (f #x2951344d0d238275) #xffeeefbbfffdffda))
(constraint (= (f #xab2e9e7965435399) #x0000ab2ebf7fff7b))
(constraint (= (f #xb0aba22681b6d5be) #x0000b0abb2afa3b6))
(constraint (= (f #x9e8412185be5ad2d) #xf77ffffffe5bf7ff))
(constraint (= (f #x624ba9c318d0b4d4) #xfdff577fef7fffbb))
(check-synth)
