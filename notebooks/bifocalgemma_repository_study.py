# /// script
# requires-python = ">=3.13"
# dependencies = ["marimo==0.25.1", "torch==2.14.1", "transformers==5.19.0", "safetensors==0.8.0", "pyarrow==25.0.1", "pytest==8.4.2", "huggingface-hub==1.33.0", "tokenizers==0.23.2", "numpy==2.5.3"]
#
# [tool.uv.sources]
# torch = [{ index = "pytorch-cu130", marker = "sys_platform == 'linux' or sys_platform == 'win32'" }]
# [[tool.uv.index]]
# name = "pytorch-cu130"
# url = "https://download.pytorch.org/whl/cu130"
# explicit = true
# ///

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    mo.md(
        "# BifocalGemma · Repository study\nPrepare → SFT → generate → native score. Full-epoch training defaults to enabled using the pinned HF dataset; generation/scoring are separate actions. Eager generation has no temporal cache. Native Python execution is opt-in and is not a security sandbox."
    )
    return (mo,)


@app.cell
def _():
    # CONTROL CENTER: choose one action/run at a time on a shared GPU.
    SOURCE = "fixture"
    EXPERIMENT = "BifocalGemma-E2B-E2B"
    ROWS_FILE = ""
    PREPARED_DIR = "outputs/repository_study/prepared_enc8k"
    RUN_DIR = "outputs/repository_study/training_enc8k_dec2k_v2_mem1"
    PREDICTIONS_FILE = "outputs/repository_study/predictions.jsonl"
    SCORE_FILE = "outputs/repository_study/score.json"
    TOKENIZER_CACHE = "outputs/repository_tokenizers"
    MODEL_CACHE = "outputs/hf_cache"
    ADAPTER_CHECKPOINT = ""
    PYTHON_EVAL = ""
    RUN_PREPARE = False
    RUN_TRAINING = True
    RUN_GENERATE = False
    RUN_SCORE = False
    VALIDATE_GOLD = False
    ALLOW_CODE_EXECUTION = False
    FETCH_SOURCE_ROWS = False
    FETCH_TOKENIZERS = False
    ALLOW_GIT_NETWORK = False
    CONTEXT_LIMIT = 8192  # encoder/retrieval budget; new pool required
    TRAIN_DECODER_LIMIT = 2048
    MAX_NEW_TOKENS = 512
    GENERATION_DECODER_LIMIT = CONTEXT_LIMIT + TRAIN_DECODER_LIMIT + 512
    RAW_LIMIT = 1000
    TASK_LIMIT = 32
    PER_REPO_LIMIT = 8
    MAX_STEPS = 0  # 0 = complete configured epochs
    EPOCHS = 1
    RESUME_FROM = ""  # checkpoint-v2 path, empty starts a new run
    CURATED_POOL_DIR = "outputs/full_instruction_pool_enc8k_dec2k_v2"
    DATASET_REPO_ID = "daruokta/bifocalgemma-data"
    DATASET_REVISION = (
        "d34a19b3badc0120755495c68227cba038819cd7"  # Full immutable dataset commit SHA
    )
    DATASET_COLLECTION = "instruction_enc8k_dec2k_v2"  # Released editing-foundation stage; repository collection remains pending
    CORPUS_KIND = "instruction"
    ACCUMULATION = 4
    HEADROOM_GIB = 16
    return (
        RESUME_FROM,
        CURATED_POOL_DIR,
        DATASET_REPO_ID,
        DATASET_REVISION,
        DATASET_COLLECTION,
        CORPUS_KIND,
        SOURCE,
        EXPERIMENT,
        ROWS_FILE,
        PREPARED_DIR,
        RUN_DIR,
        PREDICTIONS_FILE,
        SCORE_FILE,
        TOKENIZER_CACHE,
        MODEL_CACHE,
        ADAPTER_CHECKPOINT,
        PYTHON_EVAL,
        RUN_PREPARE,
        RUN_TRAINING,
        RUN_GENERATE,
        RUN_SCORE,
        VALIDATE_GOLD,
        ALLOW_CODE_EXECUTION,
        FETCH_SOURCE_ROWS,
        FETCH_TOKENIZERS,
        ALLOW_GIT_NETWORK,
        CONTEXT_LIMIT,
        TRAIN_DECODER_LIMIT,
        MAX_NEW_TOKENS,
        GENERATION_DECODER_LIMIT,
        RAW_LIMIT,
        TASK_LIMIT,
        PER_REPO_LIMIT,
        MAX_STEPS,
        EPOCHS,
        ACCUMULATION,
        HEADROOM_GIB,
    )


@app.cell
def _(mo):
    # BUNDLED SOURCE: generated from src/configs/fixtures; no weights or real dataset.
    import base64
    import hashlib
    import io
    import json
    import os
    import sys
    import zipfile
    from pathlib import Path
    from types import SimpleNamespace

    import torch
    from huggingface_hub import HfApi, hf_hub_download

    bundle_payload = (
        "P)h>@6aWAK2mk;8Apl*wQ8u^&001@z001Na003ieZf0p`b1!6JbYXL4bYF61Y-M3{WiDlSVQp}1WiD!SZ*HwsO>Yx15WOSyKUg_u"
        "X*NK`$Ehkk^hPP-f{-=!&L*RHJ&Qk*hKm2r*x5iL(gX=kR@U?J-t(Ku<JmK%Mh-e0y;Y-oYT1CDVb?*`ds;%os(}dAbTYlDCv{yd"
        "r=xMnk<TS3-&FOas_WsxBoC2&zH7iIUPRd5u(fC+u?uqhC4W=uadYgTAQz%fB~GI=uNS&OGhOIy>k_;m*W}>R4IAAv+nwu%ymVr}"
        "&M}-~pJF(Fy0C7^!CpQF<EhV~xx(kEg7iR^z^_#xH?vr^WKm&5QGUKtxI`DvFYi972Ak9?aE*fQo@R;ST7664!AT^JRqHX#l}E5u"
        "<OMqVj$ZYirCzDzShdL&lwfnF%@vFV>Yk4_0Dp|1(r%tQaJU59(u3~TaV}I3>ehSg!HaN)8`L@DlzRK6hxDf<t1n)v4+VhAvE)^R"
        "un58rA}vt`!(NC(1AT;?G*;ZUWJpN+BwBT-q($c5qiaN()9h!VlSdUjS%*2t!^}(2DYAc5s*f;zQNK}Hm{fRdd3v-60jm(#VicD;"
        "^kl>rVI{OPd!aN|xrNFt{+1omY##p)z1>nr_}xkWU4A$3gSeP{h^kajsh7Dz=v4GT^4z=%3W5wm$3d~P>tEdR@pZEd97i%(7Qyi4"
        "US@LMfnR7NE64Ke*pj#ccN4E=`gU*!F;WT}cXKlg`o?E^@X$`QVR4xR*OOPLl>c~rSKS_Lv@x@6lO=g}o$LTDldr8%a{+Y^Ys=7R"
        "k4wVUQ4o`pAh74#%Y!J`gR!VrVgX+1hG6D6h`f3`Q8KR_S@v&1H%R|)aIQRua3#y*dVhEPMpZ^zHVi(1iMoSm=0oe#dQ2}aPQHq)"
        "6%LfGy}!+Ot?b#B*y7BCD9IV_dW#Wt`7g<y5tN#o9wvCtyl2nk{0UG?0|XQR000O8001EXq%`6q4*>uGya50JApigXV{dL|X=igU"
        "WMOn+b7gd2a%F5~VRL0JY;R+0E^2dcZe5VuZo)7OhVPMh2h{7K<=7^BnaHsn6j4%?1|&4?-B(zd*e)&q2Y>$DZg#t_@_7oGx{qB)"
        "P%qQ5!F$YULKqKxI>DZxLHD+5uX)Z>d5E<E>7$N7D5MgDwZ>cQol*lv5ZXEEk%rZJuK5a$7iX&%e#VUqGl!JRf$O3u?Vjv7Mv?+7"
        "veRT;ii!=Q0rtr|4|YJMgp|=332~t0lthwjbQH!2u{P3AU8jWV_fqVo{G|Z8g;pV#r8wX;j*M-^%4J>-e2jq$x?><lbqrVa1mQw="
        "+x>5IZd1MeOL$rQ33%4emowLyiRbY8&%rV%ofa6fhI$gI?=@(RG6^XhgN@cFRNZ#-wcUIJP)h>@6aWAK2mk;8Apq&FOCQz&004*$"
        "000;O003ieZf0p`b1!9haAk67ZDnqBb1rIgZ*JvP!HU~35WO$#KLnq<)RITG)nlM6z4um%8A~ItK(>RdY)hB^du2Omv)iOg2n$Jy"
        "5i&EQr+Io0>FM@nu}B8>KrL2}<Okb#$%h4dH^qZT+u5$EJSn@Pm`JpZ|DhU8Nah-ChhUw<Ft0S|2AVVUeMR23)P*4?zKpTO;%S~@"
        "y2ExtTw2pBbobP@c=wSn$1!71Wz?x&laH0Zm&RxtW3IA#1ZeucarC~K0O0xZSnBD~ZfGT%OJnjHgh@+=Ac=U9WmpQS#*b?$G{f9H"
        "0~qX&snNfHdpE^59ac3Clx*`q|1i*WJcBzM4wA@-bI6&=1(lMCi~*PubzP*C)urM}$r1z>N0HbxS;c+#P{0tPjB)>JY<sM4$v32T"
        "bPATL?MZCGzV%*@+6TX;oyyl;HJ{rdXU&8j&{MUVx}Q>WX4~{@K7-{rPVyNX_8BmS$cnlEi6{lvhGjx(t`%XCNmif$&=U7c^b+{Z"
        "d<OsY8Bt0mPkEMeri>9O=t7h{=Nc%5QVJAk0QpfQ{x_fT(|vaN_t~}i?1e?n!f0^%!+H4^+O<8L=V7$_o#XxVd9TH~wR2Z4oWk{h"
        "?lOggHxupxuJ-d4Zk)c}-o*JAP)h>@6aWAK2mk;8Apo1sO(R_d004~(0012T003ieZf0p`b1!mbaBp*IbZ>HbUvqSIWO*)Xb8l{q"
        "R!eW(HW0oC$bS%gj=I!?dbFpab&9k-B}ID(3W1NIm$oD-B<0;rkpG^cv{u%MEayaW9v?ZJnUDPR@<kL~-Gg=7^jAMe$*V~)jCL>%"
        "P;g}dXWAQkiiPPt9DU^cC{Mpcl6e*}*=ZM?BkiKz_=rVnY}+W<vQGER2b}=xwbEi*UCnzqGg<)-_xKdnDA5RS`?lM?`|mn_t&d>i"
        "e=O5|oFwG0aF!Y}m1IT1Ar&=CSeg-;<{8fksWO`9sH!L>$-Dz12B1GqGa!xugY!BEiAXi)!0UpQN^r$#mhp<Hl;T>_TJQy<2d+mq"
        "L0I6>i+zBpY9SS8JZC)NvQm^4wBkHXQkn1~E7Dx$s;ZI=nE2l`ASEQPG6_UcqJT<9b6V$VN$@0vC=v{(OqeRwg7h|>gF27#yOBHa"
        "0V;(mA(c|-ukZkZ)1oRklPdVvtg1*wi$c-`>a{b<$G^V$ZTA+_e0>BFVDgkpjERskC1EBLMF_7e0ac{{D5a$=AcYN>_)YJ;9mRAk"
        "%0!w&LMbP~u_REkim;6Gykbmn&Y4gc?}GRDpQK9Y?svGqzGv-pNrh5nPGnUos#smp0+FOrGUu4Rw64=ENs~l8%KUW8(~Y}M`+Yo|"
        "0#sQF#%n^U%+ruRDpD#cyd70%C8-O^lZ?{pYjlIvy#R0exNnEu0vxa)uK+JAfm(r(lyh0+xX6>730g~5s}zb_6;}c3^IZ3#p&Hg|"
        "<{lNl2iAQUt%ct2_TCT23BMOc!g>8_#Zqb~jFmslp;<Nz{^O-^&1bOY(1HyP=V>fjGs;D==ESs|U8YIQ^jWfIYvIEP&bw&`r@pS&"
        "jD^NJnM}!10(O)|T6vJJIhX-@rZJg*KT+1zEciyg=IM~d*bSWPz>Ndrn1oC5n%6BJ-cBC#>_M<*+ZxH-R2lQ^L9m*JB>=w%*4`M4"
        "HQKq|MgUxUVRS#R0r}Ljb4EUby%~kjYw9(W1MuuzFlc!&_PB`F&pC{9Ot-fl#GWJ>c@SQdHE)f$1KCXP1zNv9Lv3XBiBfv%P(lxT"
        "IPK<Pk*>A~;FVe!yYDc*4`I=)CKrPm9X?vSd2%6NOK-=l+jp#O%^{vnbs_j_)QS*u7}~a5#G6&T-KU|)c+14yBguDNH=J(bkl0XR"
        "#)^G&1EvFSwYV?|+s1=I@o6^?i}Go9vA~4a_i#S;ZdzaHf3($}rj#En?G4&^h7_Ns*yr2*d6wFu;f<c#pp8x5eTMwDNMX@!ancs!"
        "1e5<~N*H?!&b0T)3e$^;-Eh}@^ZdO(+n)cH&0tzBCT<!=j?)Hzmm@U1OKEe^y)*~c+@F6{i&LxdeMg@vsnc2G!@p7YH;shQh;}ha"
        "@wYXb0DdjT=|kFd?AY{hXuP?D-aP^W{LGA#kp#uYHee{$I+)c{Yoo76#=2WK>2UuR!**EQ{T!O<A)3Sy-dkkH9&hdQ%NJh&P)h>@"
        "6aWAK2mk;8Apl(|JEmp<008X*000^Q003ieZf0p`b1!sqVQFqIWq4t2aBO8RYIARHbyBfz0x=BTk@yF6a|JtFhqhutRmC2e#5oMP"
        "Bu$(OTJ_)S+;NnX+A)65&(AOZn$H%CCNpr!&3e&92O1B-8B8kSd!d9`#S|<#@@@0b(>Vcoc?v^Q#M)<ss99ypBL_hrenpIsEVzsV"
        "YA7=~2TZ|F5r>clZQ!cxi*p0gkxoJFf%QRU`kaDo0z>Fof!3ufgR;tuF;7L{{6{s`Me^X~u8LMR;%cP5Ob`PMssm3JGxzIDVKo+("
        "#~i%7<`4?U!b<X~yy2Cvm<i%lOph_tLSF6O9yi;~_GQ_bG0V~?(gtw7h_&(=sK|@CqWeubs92M^b|fTkL(*8~)tKotX*{DYeX~V0"
        "d(fvTe*6B9?jJzxo-j?pn5ctvQ8kf_eK@|v<W^6H_*I5H@5W?e9S-iEx~EnLgDV}njyMY4@;YAlAo<*G?<kiRWW;!eQv=Y}l~Zso"
        ")TFQ2eD*z`{Qyu)0|XQR000O8001EX=Zh95djS9di2?usDF6TfWq4t2aBO9BFLGsYZ*ysMZ*qBGb98lNc`s#lVQgP=Z+CMpYIARH"
        "Y;988ZsQ;jeP5~nFmhj7Ik<^!xk;o-=}W&Nl7(TY(PE%nszj^(_X16;Y_&Wfm~)tOW`;kN)ILznaFo2-LnY)XM27_5sUquY1(ic<"
        "u8@Zr6(=T!j7`+z$jq|HkSUvCYDM2}cbszU`t^?TVPX>%eJ}XqFo?(PcE{QwKef(G3YRgd2V78>1TXRgj4Bot4aPNJN8E!9;AayR"
        "2ZzaqhA<=eXswdSA(@Q`Rz(j+;|ije`n6D!TZ^tKiZrk}t8gg<5b~OG(uYriZ|oVStIJUnLjHk|x~}UVoWlXgX4OHRx5MG6fujJ*"
        "`Iv0n6<7EB6wT`v{8=YUK{5!`|Auwa$iStb!cHPTLr}x!sB-~IGSnV(3D1_Y8A;oe5q&2fFGEU{;9_b-70`##Gbe3nuyAI%>M+Q~"
        "8W+AJNuPR;*O9yK9-f|{{|`di0+EZoEHxn?QNK%P^w&)7Im||P&8GYOO>a@UGH^eJQ(;tpZ+?9MP)h>@6aWAK2mk;8ApqrL91X$%"
        "000{S001KZ003opVQp}1WpgibWpHnEX>@OLd0%sMb!2%jQbj>TO+_wkWJQlbZp1JQ1MdOy2Le4ODEfoGVfz6!X4b+wc52Iu_3x`3"
        "irxg08c}b!jck#o8!FGz)Jh#ovmOR^eOa$;^E6Lik&U3<qrs|Z)_ezFa-4|`k^@R6B2!pOH}EN`i?jYRa6y9@;nz_1J*nqUJ$Vf)"
        "oIfCy$i?u@yv0N6119a!aABSrT)kFtl)!Y=6|MN`4z{u4R(f@RbJKe`NxfIMSSkZ{M_6>_|4H;5SfSSNdCC^Ke5~(Cx-mDd^YjZ)"
        "O9KQH000080000X0DCRhHsAmN0E+<t0BHaK0A+Y#ZE$R5b1!mbaBp*IbZ>HbUvqSIWO*-hZeeh9Xm50LFEL_fWMO7DHaIjkIWjO~"
        "HD+cwWi~TqGdVIdH8n6|F*q<aH8eRmWimKrHexenHa9jjVKQMfG&f^0H#uW9G%jj$Z*Fyu-AcnS6vyuedxt~3QJ7k5$=bb&m60YV"
        "G&4;?aw?ADyW6xKqOhw&^8Yx$bJ=bHSj6Dj0hl&lG)8<5(VtM#W5IT_-id@q>4(8kw!QGK)KytGN+^7-ibg2j$xi61(rVAS-q&>_"
        "OTMPHv6!aZTYdptmN5xIP@nsHv;kv!3g;y;i_$wnH}6xZK`3?f4(~sO9r5ANx@@|GbavDu83`oP2NcVH#OVwNv$u6@>HOZ^+AoEb"
        "qZQ=iO#~hN04pb$A^1pFq}H`A6JA^XF|I6s`G2AEc;9ZO`2$c(0|XQR000O8001EXyq0&G-T(jqiva)tX#fBKWq4t2aBO9BFLGsY"
        "Z*ysMZ*qBGb98lNc`tKrVQ_P3Z*+4nF=b+8F=RPoWHB-`V=*%`HDxnmIW{#lGi7EpH83$^G-NO{G-hIDVmM(qIWuKuF*ac^W;tPI"
        "Wj8f2H!)*1E^2dcZgr1sOT#b_$KMb39gg@$VJbG-uKOxh#x=drXqtrNDvsfIw`n^>VXuzl|8T!MZ#MuUQt;vcLYprhBRz)b-%0Vk"
        "6uVh(G0{=^VKAI+m99#ZM0J5!%2wC9CRr4F>eMr~`=^(p(nPXa)7n@{Q|>jt0M5&pgn-=VzMgELm>$AuNz9`1j>*mYI?{k$cl3^K"
        "KhchKcW7KT-9S1y?um^8#`FQ@@*inB!@=xroLD-)ch~kSv}&}-K3+x8(GRe4f*FF3d_ihl<1*o;<sak9@|XV?Dv!7AW|}`xO9KQH"
        "000080000X0Ozjx=HLJT0E+<t0BHaK0A+Y#ZE$R5b1!mbaBp*IbZ>HbUvqSIWO*-hZeeh9Xm50LFEeB`F=8@gI599~Wo9)qV`DgB"
        "VKp*kIb<_rG&MIgW;J3nFf?UkF<~??WivQ4GB+`1Wn*PIW-&N7I5jdkHZE#&Z*FyuZA-&25XavS_8pG+Mqz5VZg%dgSQ*#kLNn7O"
        "Bv)|^zq?J_AqsnSB>#u|-DSA|5Rrlx2N2qPaUbb9M1La5k880S^#&6irSCe!*|x6KrqEhpQI(C{Z%f)$)HH;<Vq3h(&Hh!ADodQx"
        "+E_|M?k&FnF4Gu<fZXQ3mTaJyp2B%bjH2|8$&LFIX+Wip-qHOh+MXT`wacbENN2|_v5~--KA@cbBMoOb7`?R<Q|I^U)_#dr_7>U4"
        "n+Q7k4rWd;UGR~wNVTh7CcL)%W1Lz3^8Z5R@xELP^9N8%0|XQR000O8001EXE-TfY;Q#;tiva)tX#fBKWq4t2aBO9BFLGsYZ*ysM"
        "Z*qBGb98lNc`tKrVQ_P3Z*+4nG&EskWHT``HDzKnF)}tdW-((lGGaC`Ib=34F=Q}hG&MP9IXE#gHexnpVPY~kFf=eYI5A>jVqq~k"
        "W->G~E^2dcZgr2{YQrED$L|Y$hl9J>pzW5A#iXyYjH2d9?2L+<u~PQ#i%H7ZpjQX|f1KaBZZ`nLqqF1)M5?bidVF@?o>9boPIj}-"
        "S-_L9!=Pfa<wEionw2!;MMbIP1$(WlrVvd{b6$&*WtD6=Thmf$3{&ncy#TJum;}dSpZa>#4t+Sd%aWKyVNFCc@9R(k>$;OQ`1r~6"
        "h)>7XB+~<gi-|o(B_Oi!0p;@VaXQ1m>}?HMI=^@K_6w%Ps9CiBE`s!SfRz)};H-}~q}H@15#Cz<F|I6s`G29(c;9ZO`2$c(0|XQR"
        "000O8001EX7KqqE-v9sriva)tX#fBKWq4t2aBO9BFLGsYZ*ysMZ*qBGb98lNc`tKrVQ_P3Z*+4nH8?gfV`5=2H8DA6HDfg~G%+(Z"
        "W@ctJWH4khG-fqoIAbw4G&49iVK_N9Ib<+pVPiBoW@0&IGh{h2WH>ZAE^2dcZgr1KYQr!PhWCNoVNf?Mq-jcuW#uYy33`mMQdtu6"
        "xHN>`y_W4zTF7dU{>OZCU9SLG#uV8Bm~~$?WPC|Eo>9~Do^2*w$sSKScAcYQQ<YF$z1C*0cCr>qSEvNHTncLqH*Yn^ii@3D($;zG"
        "$K0{J0Iu^Gg(RsheJ%Qgx!<PCoR~x%17Vo<8MGjcImLjFpVANbbZ9~`JwSg6)FL?zB>NBG^MA(ii~y6j32g5C-rU=-rPqU(6!TpK"
        "n_~wHC%7)fOgE$^G@%gQTK+LEEPwfbp~`q)ug3WUP)h>@6aWAK2mk;8Appz=2w~s=004>s003zK003opVQp}1WpgibWpHnEX>@OL"
        "d0%sMb!2%jb8ca9b7*gLb1yYGVq|4BVPR%vF)}$YVKQW5WMVL5WHdN2Fg7w~IWuK4IAUWrHaRmjW@BbCGcq=2H#9LZH)1g}He+RD"
        "I5{qAb8l{Sk5Nm*FcgK~2m23~_@Xd%){?dRD^|vBazle@5|UIL!~bs5c8J2B4#~aeobO(?8vqu_J9Y%7^jDe!Jr88$qni@j&Ep;;"
        "orD`loyyh=?<-YTO)F8-YrSt#@?P|)>RQPI=jzZjt*H3wR@zX^zPIuMxGa(f9@9|ra+{|RpZs}o%tPU9B0Kj<Qh-vPoTd9uG!s1>"
        "J6o3SAf9a+C}{x+;|J`Q|3LW)3-fMg*^>FayVWnyh{<4b;U<C#ZiJN+^ypnkSESB%wh&%Z{xPmBfBAo*io9<(S$_afO9KQH00008"
        "0000X05d!75#Rs-0E+<t0BHaK0A+Y#ZE$R5b1!mbaBp*IbZ>HbUvqSIWO*-hZeeh9Xm50LFE%zYFg0a3Wo0vCIWjUiHexd|GB;sm"
        "I509cF*7%1F=jb4V=^~oFfce|Gcsf~IXF0FVP<4EVr4XAFgGw{H!f;(Z*Fyu-D<-i6vyuieTRd)*`VucYAx%lETdRG5;vn_PRdHz"
        "yDugwV}o8D^#5^w=ek?~u#73P12F5psL%M4a{NF|&l|QH^^@xGsAJnYDmJwfLRnsIrL45%3O7=UlQ4pd-Ck~AO(oFmEuYiYdF+PV"
        "TX_Lor!fdgQB(RF^a*pfPM0Y$iaG|uFzz#GK^b$50UyuG_xN<ELoq!-cL~%WISr(`PvFyk#^H<rqqh!h>ik~a+pm?^y;l_TT?CtB"
        "3o|FUHpNUgq&n205Z+q;G0rT1`G29xcwa7t`2$c(0|XQR000O8001EXb&f<(-~a#siva)tX#fBKWq4t2aBO9BFLGsYZ*ysMZ*qBG"
        "b98lNc`tKrVQ_P3Z*+4nVmUT6WMMfmWMg49V>L54F*z}1VKg^4G-P69VP-fqGC5*5VKFu|H8C(^H!(S7VPi04F*G$cVlg*mWMN`v"
        "E^2dcZgr1KYQr!PhWCNoVNf?Mq^X}G%~k3WWO;<u%94=Ap&|6{wQPsdLRN$HKjxe3b_2iyddH5yl>VYI;ENCL10_8dY&Ywzh<K82"
        "7&IkYCKVT1Rvl2}vaIVOFIvva3|rZim8y8jU-Ls%uW6+X#wqufUI5o+OoA8Gr@kJI#}N1avLt3vI!kEheH|(gU3YR8AJ4*!_;hS+"
        "GCe@NSn83q1QPKR>g7M+bcTi5+gP@Ae(&z>m%_-=2y)>rf(mYcl@s*fU7#CMV;h?YZ!P~ASC+s0zfftsZ#UEY0Z>Z=1QY-O00;m8"
        "03iSk$`O&^0001s0RRAL0001GcwudDY-MvVa%FIDb7^#Ma(Q2KbaiBTFLQ2TaC2yHbaO9dWMVlvF*jm0HDP3AV>B`_F=AvnF=IC}"
        "GdM6gI5ssnW@TbBW??urI51&lV`DWlG-fwsVqr8fHexq8HDqKiYIARHb&uUj!!Q)b?+1H_L%dO#+OBhrdlf4qJvpIqX%doD9K(0F"
        "X*)z=SBK>Paen8rUIDN`-mwEPrN7b;=-CJNL6{y(wwd%+L^`7Dd!4dvkCMw*tV&+;Es3J!FLhOwsz#!^Mcx+muBiBKNh@tAj=8t|"
        "0=Ud$6ud}X?(4{S3h~LG=fotUvx)4qPm%&5<<VKX{}g7RheKnt=?>!ArjC+ENFttKJO2k7&#*9g8_VX-@6E0KC=3opBo}TXsNi~7"
        "I6?Q`g>*$~Y-2OwwdEh<!t$5@7b=hU^=h0yP)h>@6aWAK2mk;8Aplm-+vP(7000;d001fg003opVQp}1WpgibWpHnEX>@OLd0%sM"
        "b!2%jbaG*7ZeMb5cXKXkb8l{J?bXe0+b|3O;QN5RgOFV|q)zo`O6n=Q>?r~SlteY&9LMm_(4yG8Pj<VZD~bTQ3^~{b!xm|h!Ux+="
        "tE%Y-Yy$18xVmGjR6Pu1N5#-#)f9Yn!&V)&r&podwmW^NFgyo6_OuVD<FOj1906V?Ev<_fNonpdf)5^()+ri*WUO^T!t`ja<7+zZ"
        "bN_*k)XW9GRSzs+9mX1mQ@8nZNq4LoSbsQ;bz~joIT|(`i1lOJ))hte6YG&!TaP_EHFU$qzH090lmpc*b(^^(4&g9$iB(FhDDHDj"
        "ynRmSH|-<yuWoitQ564#cg=deev`A^vO?S_ww_Mg?XF>w62lnxd5q8VRA27$>U4Z!{bQ3y8mhns?0+zxdrX0s4b9oP$=fBiW@6V&"
        "8>p*5O;6JnzP1<+RLITD@VD^(MJcBY-AKPaet0@hEG{L+Qfo{?DgYTkAaMZgb<{$a29%3ZQ#45|#1pZ&0%CcNSiD0FA=>DI5>{AA"
        "$b~J9QX-i^WPGrm8_ku~3-LrOuYg!xN-Rv2Jf(=-s4S`#m4>`0EV<?o@)tZ%R0<d3iCA3$u|7vE-yxPE<rh_GBfJ7*l2=$-a4I-!"
        "q_;t%faLr_JQM3HATBQ@<{<%ct}USmsw}0D9zb#OAu63TC~(5+av`3G%PSy;bHwT$Vv`J@B61QDlus@>Qc_xq`33e?l+q+3WftOz"
        "7_NZWTuSW0dE<qR%5${N2O#gVJG%UXHc?wcU>uJN@kDI?2k~!EO9KQH000080000X00g{E!)5>g0BisN02}}S0CRF<FJftCZ)0I>"
        "XJu_|VJ}}_X>MtBUtcb8c@4}l4udcZM&TWaJ6PU{=vs-T13dujk>&iMkqD`rl=C<9`}COkm10pj-^f`=%T=5Gw54)URf(~j;YlGG"
        "v`yQ(Q-&P~9R^H>uOuTwg_wREGmpbKcz?rcKX^|I@$4RM?%@wmO9KQH000080000X0P`loJ(C3h0E`j<02}}S0CRF<FJftCZ)0I>"
        "XJu_|VJ}}_ZDDC{Utcb8dBs>;Z`(E$eh-lUK<G&tY^5mrl)(d<#_58!Nl>TTfGiLg>B!<Dks3)Qal8Ka9g>o*o0Y7<Fu&O1;kkY1"
        "Dtn%HnWj18Kw`m_mQ}2o-~_mqn^G{YRY+C<BoVPXI{AG5;WT>x`TTSl7D?!N-r>RF0V_%&HKB4{QmJ5j@TU^|@F10<APLnp&Z$zM"
        "NON5!EY>x>q&mylT|>LVM<dzN?pP{fny;ZKXoyCpY~9@R4-W`_mcL&uKb?PEUR*~PZ$B<qv%&G|YWc1?zWQ`|cK%^|YYjSrrxGM9"
        "(CPZ{f#wWth886%sFv&rF9c(T6(WILg^V-JkogO>FxGY^pkm4HAnKCG<fxdYsmPOt)j@vp;qu)ts~I^1sw)W>LVp69Y?e|AX@ELB"
        "NFXHzW!%RwJRFmpDSd<Cyd{5=3xUyH*sp-he<B#vW9O!gmqc0<yYQ`CSH{~bOR4cSALl}8%wBjW&f@pR;%?sD<s!aUKN2not+YVQ"
        "SWGN7m_d&`7c^0!=fno?OI3GhfioqfNfP0Kds04X)eA2csD~%sj7a#aViJ<YwXC4#XIzyKYm|YERzTS|-mCARgbea>;jJ?9Na76Q"
        "dz<5=l;Qz6<uMRcWU$Y{z~G{&@Vn}pU5m5|X^ooi&F8y}L2Yp+SPW`0$WX*HrB$v6l$%$#K{p47j(H7f?#f~7)~^kV&cL|`$(*4B"
        ">}l?T;F37>dO8MPYz6C02@6w?JCn_m@3|Z5Xy>mAO~rdb6W2oAj~Q>zw<cdRz9DME(PEP-5>k+0;*bjYNM&+#N8@|*g9x5)wuW0F"
        "U8`H+-V+7?hXLBhZ*FJAqj~NHlQHRL>`*zXOyB{vwAFcWS>dx6R|##)x97Dijw$xA7pA!*Z_&DNx6>-mQD3Puq78n1tirDJm*waX"
        "ogp&rc~bWG!JsAft`QXM`PThY5nDZwn2Js6o>S&Dh75Zq_K%Ld|E@Lx3#S^d&4x5)v;>;ZGb%YKHEMRlJg~gPrpK(is#}y}9T6GT"
        "uBF;9BTU)ZumxyyysH?NTBfk>v}Cc^^~}~aoBVi)!n<SBx7@qupvzNA)+x~DWBNsyR7I&qi01Qk{MZ(GE11XDs!)r_faC|W2{xCu"
        "5#~?Ia!|-Sp~NZTG4@W0H{H(SHygx_)moE7#lFC8Ff9vlZu3lE#(e_p)Sht18koSOfi<}{a6;I(DcriNON80WcMA6`IKI&zOoqcP"
        "3@GWQ7G)0t)6(2BfrGZQ1l>+a!!ogz4@vY0Y@KNpnUw<)RB%QS&ppihJZC(?{93#oA@?HPdW3weocc3+q^wNQ?%zE9TBP9~t*=%!"
        "eRb;4Ytt66_qz4ez-&{3)51<zSIuTaFzRo=1<$(SrgzWyz3cF9oY(8Dta!9NjI1!Ty>^g*9NP|d%LN(f3jL+tOz`2Y#_}DA)_60$"
        "hL>%f`Oh@8+5UK21lm~E4jd0v4FWZShb@|!v)>KV;DupnwuPSJ8hR~cob|!p8@YNfX@#wBoTD+bvO|Rs@~Vh*hJ!9qeu1weZ)P=d"
        "-6M9K`uB043i1CQ(Kb}y6;8MIJX>Y<8slsC1K;H8aIMa+-`PGfbBhp;MTrhXD%FF?_2?teKZoOx!@VJxm;NJkiKX}!XWtFXE;+ot"
        "wRUuHJ6WRAWu?`C|5+rr>)5kk);Do%h0TiA!1>$q1nV>V*&X*G?2-MT5d{vb*6xO#xZ+JVdEKXG1$>Heu^)ZY$7xHz{Al@7FM069"
        "wEucWQtzw1@$C=Z6X@g1VFgysj9jklkKjcT&$%5YSO^#bT$lO^aOPH$lUMGK2n<3LnfVe$HULp%{+5WMCIFakSe&bk(oigo4*Rc#"
        "DT#joP)h>@6aWAK2mk;8AplujE6q;;003A3001HY0047xV=rQ9W^ZF*Y-eR{ZDB8AXJu}5FJE72ZfSI1UoLQY4adO=fG`XN(7WIc"
        "A$MKCo12v8QwlaIX)3zCzT0Ew)20f91UD}0@GTW+P}M;36xEcv{Tm0Vk)zwFVg6FK){@Gg<RCoAGax%NugeEeO9KQH000080000X"
        "0Alb-By#}(0GR>+03HAU0CRF<FJftCZ)0I>XJu_|VJ~54Wo~pYY;SLHE^v8ml1*#eFc60C2LFe_rv$v-^y))lX_MYc2)&nCY2+2v"
        "7c+9>{QHcy8@6eqn?QQz{g`1G4tJks_(n0}P6`?Cyn&i?fPySOEr7`algVt!0-EY`f{=5OX&8pS>pRCe!qK~IFf0u;nVc%?Qvn}K"
        "<_b?#(DVI<u>=|~*GDGn&%BQqV$iu9>>J{Nu&zwsbA2MTFZaxu+w{IO0Y%^|{icu$@nNA<=e`3ymK^Y!ZiIpxf&R>&|CJMQ`ze)*"
        "%-~bKh27`>6kd4ksOyKdNtrbkw0j|BbK2|6y00Yll97zy{HBo|SG$l#(Q<6T&{p-MW0KR<<iBx$w9pquHP0hdA1SYwZ_k;3NkKz9"
        "F>2AO6Ef?$iMZ;-Bej$QMRiVgx-t(TfcQ=4My=C}%7Rr0>WM%w5?~1VS%Izb_YyTSIvbhZS-)Nwy^=nv<)$swFv9R9r=$?jMu@S&"
        "#_MhokAL}Squb7quGMYdy#i260|XQR000O8001EX1(i)TNB{r;N&o-=BLDyZb8=%ZVrgb?V_|G(Wo>O?FJW_YWpr~dUtei%X>?y-"
        "E^v7j$<YabFboC2f59Cd8NdZx!W9Ay8f=r22Bc0e-tQ67p=MIhB)M=h-0Iw6jvkW1`sX8hUOC`_^jl0{33Q&-xr~j7*5w0GO9KQH"
        "000080000X0G6X=OydFo0AUIM04D$d0CRF<FJftCZ)0I>XJu_|VJ~5Gb7gdMFK1<KZDBNDWinzeYIARH?N&{1+b|5h57>Vwa&8ja"
        "iPLn)0s9rhAkY@=Xq6>TqTOc2{`)9@)p5Gsw!_dM2m-z*k{0Fn^!)L|&CRN^617lr_336+T3c$|mRMD=-mkwTpVx|3w;j*h#^op)"
        "rvvjuRQpfc?WrkCWy%6_+~juqYJ#mpZ*_~?3Y6LA))cDTeD~H2nOk(KP$&gQICT;whjkpSqjbO>w#us>++mXJ4{>ow@pvDnyLi7Z"
        "jv%wU2zS{rPxAc!xKH73SlZN{96}|?f$@cP6*@1t;FvGnq?>3Xv5kfmgij>;9vOcQF^&Qe8saq~Q{WjWKzqEsLZ!2{ZHNicCc1_n"
        "1pW~1a(QOGAp3x(pF%RmC7o*wO=g9*K=ENUB4{AfC@PC0(>8xtX2i>k@f3fm(UDMwkat2VAQWEx3`I<%^++p}0WGvG6$x((%Au}#"
        "$jsW!9;?(gW7-Kp3aKha(|GEA7ERK_6jf4kQ;?Htfrb&3s2fqA$-4X`2+V1<pbs7Lw3mqjBKVPnhBN;vQ+InmtEh}31)6zI@DV7n"
        "A5HRtUvI^M57mhGiNhem)W7!8@HS{Ac}v7J+p+1tgcUrVe^%_hk@rUa$42`2xAEPs;7Qcht4^A+%20+x;XRLcadbCnMDLKJ7<b;b"
        "7r`ecp)yVNQl?YHkmWdsd+GHHr47iSqYro%E$9ZP^-JO)dYvQ%e3#DA>-F&k+zNjJ$1XdqPbMj^S*&M%F~wcwg<;baSa%6=9Zky0"
        "PRH5{4l1NR-}9gLeFk;KT##hpEP*=<zB;h<ce0$Mw=-T>fKEsdN=Myia-SJvm`fe3*M)kZS0m}yTj%7mt?P1c6q8s$qv@qi|7R>W"
        "6?;@h+Q&<4B!%M9I(d2I$TgFFldZ#5;nfq<c=miw?lAA4_2Pk+%pCF)F0zNXv`x4`JSxez2wg82t<51D-7q;MGe3NRUi00K8koEv"
        "{QV73O9KQH000080000X0JyAjh~okP04ECo04D$d0CRF<FJftCZ)0I>XJu_|VJ~5Gb7gdMFK1<KZDBNDWi(<gYIARH?N-Zf+b|Hk"
        "7wA6(KF7ACNKTq#fqq3%5G!#dv7tznq~y3k|Gi5+7+H36E80VhzzB9`ALP<I`||O_&CRCK93>Nc^LVqVw5}xXD{LCrrrR&+=dEy?"
        "+kt0wX9{GUk)ByQ*V@PZzV50@sH%hlcZJ?x4X`(8wd`@<fKdBFt5Q_E?^dfRa*sxo0y!fKg-)X6u#KZ_lufw7URW`Lh!K{DD7iZ%"
        "aZ#38l=A!RAvq*G#VCFFltBh%3d!W!wbmJghB*VumfAFEETe>Tylj{4q8-ORnmimnJ2uux_=~rZ6`Vum$cN<uECD4*i?<gjH@elG"
        "BSN%`F5xGJKUlwZ&$VSvKcO1Ckj!@OPA&Q(^HOSX@nJI~s34cf8jUQMx_DYg#Pf)8us^qG923eR-2@llkXiB5CsCc0Gp!T~B$K)l"
        "j#*v00@{|EOsu=bu_~!^qSXkTanTT(#i90vHOUUaDmbU2pm3rGH?^Qb*|GNIv?*%lK-HTSeX5Y+L{|_-M`99A^et37-2Eb>P{Ik2"
        "#50VZK!sy#l2`olD&i<fL#8pZ?}Tsr*U{>K4pMNs6Be2~SN#vLl6v=F6!Ko=dy(%&{@o%+`?q;MHSo+@ZN;E3*yPB4AU~_(yEwWJ"
        "gKm`hSy2XoPMH_GFl|-Wyrvn(x{va(gh&0bE4lE<BnD4-Vm-(Xga4X1P3~c=dOS^*=%xQT+%sE)p;aOH$DzQM6lWoNVS<~+GDXVI"
        "upJz58wGhFxh4St0}Z+{-pilveE|*0jMo)rv;%WuwEvKP4cd}qw+r5oz}4tr5SHN+By_=;`qKLS!T5Y2H#6ziTkqt&tjlz7B;!~@"
        "C*5nF@t-+eNOly8>*Kk;JAv$|4L^T!&U9f1ak};U&ea#wSz25uxWQsP>)C<U%mRuUSJAzl>&|aMN5N?olI?m4>U9BmSRQC!Ow&u~"
        "+V(I|J(Dhk{(b{cO9KQH000080000X0E6usgdGC_051vv051Rl0CRF<FJftCZ)0I>XJu_|VJ~5Gb7gdMFLZBfWo~JDWpZC_VQy(="
        "Wpi{cYIARH?UvtaTt^JX?*n-U<LfxpztN0(8R&Ba&5Y!3YI}pd^r9i}UM*?qlGY`KUW7L6fnCm=BWXVU^v!;F`Qqy8`u2Fdy^HT|"
        "<Nn@n<mLx@bN&6*hx1Y`KE}_7r^E4~Ze4G(Or6pa2D(@Tb!H#U*7-x*+D6yBe)ZeSdAN(GKZ4br)A6TxIQ$Ziuirg=3W_v+*`V2t"
        "m&gR(jhq^|C2W|gI*_+F)oBQe#C1EUwcLoIzbCs4{&pQ7@^E{Rqc;f_p^(1NakaKvtKiskN%tV?F|cWDoVWw_5E^K$>zHj_<oGOd"
        "wDhPtYu_VDxTgX_Cq&UXnlOjEL1?#ntm+-Wt7k^N)_IPL$hYwjkA5nxKQDg=YC7)TJCD9whVE?Q*0gO~kVwecCu5d!D@6vh>WvPq"
        "<$hWI`^V#or>EneBx0I7rET&pYmqq&qFwuH-Eq*4t}2osXs$$~7Er8FNoLibb^R&D$Me_6SAWTsJjazFx?`VRti)=x6S>)@y}~;i"
        "c5fov{Hx<i-FPrsPXO4|iFHWiCPK(IlWJ3?WuBYD3Ld4V5a2B25)_{2DycivpfN4BoVK<x5Qq`%&<Uo6C#(ds>eZ;<2ABoa>I0JC"
        "|HKu&xjCGkL5pj$MKQI-Fl>yKYxtfWqNZ=rDnz7t|3_%8H8wk}U=`K1R{6`AX{BMd?6@a(v}&4{Sz_%{r6gEWsm~0^FG0&vGM1DX"
        "EseEzj;zK|RM7Bd9oN}Zc1Z-5oCc~+Gh!@OQszsve)QAZQ{2D5^HaQjcX)W`=L}FS9ng7=KD_ccfK&i0M>iV^+PKeIMFT1$aHG{}"
        "w8iCPtIhU2yf|qj??M*=Qdcu6|D6NN8Ov7Z?z159o-%4?N~KZRqXs@!jT3(%yh5aN1|#;e3TixE6s!|WAGCDI3Mt9G!191rh}Uq~"
        "U|;10@s+?Ue*<p>M4cz)DugSuf$ZLjCfX7p!<siQD(D-$|02B1vy2t$5aLBK8f$qs70tFAm-g;XDRo5b8KqN&HSU}<45<G9$bS>w"
        "moMu26;Mk91QY-O00;m803iU%U1_O~0RR950{{Ra0001Ua$_%IX=ZO@VQgn*ZEayMV`yb#YjAIAZggpGXD)Dgjgmo6<1i3~@0I!="
        "Mmd2L+TCjodq8Rt60is)u4rmcnh~)bjVE0#zkbI_f!!?!a!EWM`@Q*Is;YYY_Pv4Jp^%3G1p_GwWD>@nUr{+l!04eJ`yM#5{#73&"
        "9VlXSw5qCVxmYe*Nx`nsuIC+a7?LR5+wy&H$8B=h?<oXah2*Kv^Xskqa&EXP4Yy_EvxC`T+os;X9oiD=D0G+ffzbcnh$JbJ%Y~;F"
        "G9KvJb>v=##8J;Cz2;P@hBWFh>a%BQCfnz5@dNHsq-MDQdu+<>Sd<!YDYm>{=QN5V5IS%+Bzdjrg(ANi!wz#txd!w&7}GpqG<cZ7"
        "AX`;&##U|?<V>H=aaJO#YN)Qm&{LqO^CcG<TY{{pR3_xOp9Fl#FIZ4n*_p3A221DsgNBtf$v;3oKK~p21kou6q20)q1d8ie$(8D4"
        "M42*NZni%kHjoD8P@t@5zq5zwC;AeCxnWX1!gw+Uej2^Qf$H}FYWS7ph?!>OV>Bxs?r+RTw8lQ6UmZ1#%O6ln0|XQR000O8001EX"
        "hr*_*d<6gi4GsVR8UO$Qb8=%ZVrgb?V_|G(Wo>O?FJo_RW@%?GaCwbaTW{Mo6n;10|G?_W-fY*BCEtb&Yf`t%(56A03<H8dpv5Db"
        "i!Po-=hDpoeTNh!Q;OTQd8o;AcrM?$%CfApp*<%7P11x1G$J{F>c=UAh-9$lxyYUeWC=i68su-^U7h`Ye!V!mzPVZqVn%4f$mz}b"
        "r}zHHoAdYbYG7GbyVY*-c$;Q|{GF#syA@_>Oc)h3h-jWePBfL~j0d83AZgYu6`P2!O14WpYKnN?@?>2SK5+pVjikQM(&Q!o3go1`"
        "zO@$4snsLa!acR_BvZTfTcxUl^1ecHa-Hp<+is~la!P~yRhqzMmWDindAmjMUyfquq|#7aZa8C*_yvt8qPUIdGi14OasS|NfU?F_"
        "zM&Z~{}uO&=HO@Wz;m7^bAsp>_goyi{bvW+vmqW`pK3Ew45gy@jK?L>@x}#t9i^)ybkBCQ!x3i)v;W8wmOj2B>*t2`NSze4AWZ~3"
        "2|wa77bVRe*~5u`{D7vdb`eF(E9|6gW5;&gf@K>4s%B#I%5awMRs|9KINHw1DorC~_&}o^iU?m`efe<yX|bFWnOnD2)OyX}-r>8K"
        "Laa)Zd6S`25G#h>;cR=Ov2h%ECe8Tm42^@uoLLx-SFRThr#5wG0M2AIqEl}c$Ule8JZ95zXlk-X;5hEs4#yr$XQ&rZ3a6B<CPNsl"
        "rhyX#v#IORiTS7jlge>T?)FNT-T!mht+|XbFr9H7lV-X<HLZ7Tn^JqSnz%DcXP(1Cd*scS!vdg_(afGu&vn_1)?5Y{LO32eBg}~)"
        "485VtCf*b?gt;(ur!$XwG<2zB89wcQsJ_oO5ZrH5%&IF0b#<#|=W}Kh){$B}+5@r_d&HsLKoTH5^JNvwzQdq6bMb@iXYsU!jK`3O"
        "<rV^MUqWLr4ADyFbzjK8<O0K}s|1d*6f`OXibvmk>9=8g4A~l(G&`WXoJP`Q`9N^+OUFuw9@9*=99evtDWg1*9V3H)Z%wg`<CXC9"
        "5HY}lGtb&soH0C#s8D#<>sbLWTcy91KsSrM@HHeN^<}Mha)`n%>Hn%Ke||`)48$%=^t36sig3usXmZ$TM7KwN>K%ZVYL(by*|RMQ"
        "2hA!Q&#e@8S0<9uTzhELSZJ-&bREIlHN-LPyZsmWc5jiNNyoAX4@)Bmpv%PSV<vTtO3}&(wXyN)S;=J#5krR5SSvm=iC>#w<aAkm"
        "29{gOvrYw^foGMH)ty50Xu@mI>a-ear#zGc?xRBOMeE!5U$mCJ*-cb@HA@xLqU@Sn`uTxNWtz!WRmtJ1gy-a>bh3`i;;+la)%oYe"
        "#kIe@T%0X2Q?xGyTGf(*|COprid(HOO^WkPiu+1h14!$mPB!G}pI;-l>>RxU0xh`#K-Cj^qz1`1CA-HYyFVzoTLbOl-RFe_P3*7-"
        "baK=)kR6ePX-4D;R|L|;*htAW2p=jlJ5@9NcKUr!u+dboG7iRI@Rj5P*+^u`rZbyztf4<?w1Y*KrCBGmZj$>XeN4!~M)D#>y~*|i"
        ">>Fu{57g%S(!^vxuna#RDDG@jfZ?nkQ<3qfj@+h|&p0IHDy>{saNgDd=#jx-aHpNXJzljfA~rw*`3qPdA^BtZ<$};P_WN8YTKSpq"
        "3rXMs3n2s%Y>F4E<>kdEf$CTEIZ7$RXv*(`f&86!q)E!g?jVy3sazk6g2jPyV(moOpIY6no_=U>Ab=fqH)U&%>SF4ndfDA@uQ8V}"
        "t1+FBY>uJYhGq;SWK$_wYmOCOIg)Qx0+iFI``ygB(_;2|p${uC-kelknKP@ID%K%ZqzEjIxiCbJi~LYws=ne1|N6@h*N4wIt6T4{"
        "4GY_?{{T=+0|XQR000O8001EX$R$*uFarPpdI<mk9RL6Tb8=%ZVrgb?V_|G(Wo>O?FJo_RbaG*1baO6nd7V~mYuhjo{vODG5dP#q"
        ">$Urs(b0}(5ZKzVhA~)&QS5Uoq8Cp(cWv2^-$}L;yLo9kLkK#Zp1XTa_t;SsU7nndna~Kf;2Ai!0Q>`>S!T3DTV*JWW(#SurAyci"
        "0LxA0-d$eIFVfj`Hu*BWnj|Vuq9_^;hJ(Ty#d3~3liWFi6RI>8*QO&@2#|T>S#rFWV$pidDho5?#N&+Y9<H&Yu~}FT2j?rKm=K;p"
        "pUrIrqv0Uf*f%KYJo!I;781t80i(x988fP;UQ%TU8XHS(%b{RtN@p<@T40*S4rDQE09p`s)=x1wC4OhLb`(YMO@vGN!Wmg1#3N=b"
        "i;^TFrt963TXeyCdmF4Ps9+rTZJj*)W4Um^zH?c@#9CwHXkB-JcT80dsh7-x<R~71&8}~zKblRUoydXVJus(v>Fy0C(ddl?-!vzi"
        "+$?Q$NM3LUX=aoX*s&JW6gOsAl@fmTdv(N;B>8n{4qVvDBUk0B$5aw>ok1${PCRESn?cu5%3eZk`Qt5xHNehQT?FLz4lpGmv-(o8"
        "zwF9rpu6G!4k}=7mY~Hm<l8EtNt#(G#3O}fTw37y6PwPD3u6^Wg#1I3L%7JC3J;CUB&DzK+CB6k+Ns=45a`V0V9B^@3u!^yi+VC@"
        "k22)6$FU)1w}S%Yo|iGpMTW#zP=c67k5>{?k{{3hncv*Fi!F0E@cTXeJ_Yt)l%UJ14e#|LnS*xbMV+2N?IkT$+%30xw)O6XYn@()"
        "lY9Yk17j^ZXo83w(c(cgSCwv5?OxkP?D!*#x|K&AJ+Oe4)jeY3zFGto*<dN5$PNC{9{+xYuN>Oo?m+)H1*sX`?lfwtJuj^(!Q#`J"
        ">9(RU<~|nXSF60|x7(jhcH@1VNTcuKQIb{4RVqF{B|LUYYe`d`Dxu?(eQB-c>%Y_5!Qz2wZJV;MgeR{nq%Jl=rMVOM;u<%?*E+ae"
        "+XpK;$(|p6s$j3EE1zAex$4UDT8|G3IC-F#C-<Se?%TG^1-1BIal7mw+e-K2TcEf41LWbesWy4)Rf#>D(1qpoClq}%kh}>=|E}@)"
        "2T)4`1QY-O00;m803iTX1w2xN000280000X0001Ua$_%IX=ZO@VQgn*ZEayMWMOn+FJE72ZfSI1UoLQYRgFOo!Y~X3@0I$8l^@U#"
        "AWr;%D?*V;T!`evmE#M)r$t=2?(ArY=qj6y8s={l4o0>f&N&9S7jWS%#K#FPtl!6wz$%enwLeh@x&><6RGRdVvJlZc&C}}E@TjY0"
        ")~B?_%ONk>RC%u*ig{qqbM?z_J;an)H6@&Jqkl^GdHMiQO9KQH000080000X0B64)>bn8}0I3B403iSX0CRF<FJftCZ)0I>XJu_|"
        "VJ~E1bYU-HWMOc0WpZ;aaCwbXO=}x55WNTTABOG40ed%x(1S}0rgn-UO-&QpLn&b_jl3dSNl4l_ru4sev|fLOP$Fa2^3KPb_nuS~"
        "#q{!O3@Fhj3K0ScXJZPjNZ<pFlzxV2(Z|}QSi&6<NGD=7o!n0z7W2jIW_~*>8(9`baej7wrh;ogIAcg8&6P@A3eb&rA;FwGU~-|4"
        "@|bF)m%FzM-tH?bwQ>T@ikb#XMw9}I6nCND$)gUlXKcKo*D|6&rJRwwglodo2D)|WJE7dnLdYocM;MF|rkF5NnrvAZpPvD*=-MC%"
        "K0?u24$>=2vYRo#l%zm&>;r94#6cvZhR~qZieegf^9&(+qj|%OUcl(no`PQ|BmC|==EXc-w;AjhytdqOhl8^PT4yy+dCI^L);iVT"
        "o+udfdkEzYYhF_a)+I=RnP7dIMki<}MqE(@jKH~CY3{acR}ox*C0S877#<;ESW(JRhjOfkw%V>hTi<45OhjrKm!c$U{=Yi%5!;|r"
        "+VJFELhOL7bs$-ujD6T}@kp<VHaG-zS!~{72&$N89vZR<{$w4>EoXlrikcc^O1#g_(aOD>;S^$d)c@kx8+Sivcg$AZn`3b5m5W}w"
        "cW{&K2a@O7j$C7`;R%hFnD=?`>E(MS4C0z%u2}lbhdpBqr(sTW&{g@;fQ(_(2~y=>U2<`Z@pC8W0>$I?{r_<uYA*FkI7!vk44!jU"
        "Dqal!Wz)R1Tn5JBRsTGxG?Q-w$0qoul+l)e5AQ=h_bh3w^g*V443?xEUrou4P;qv?NJI_Mu%-iM?_zKK-t;Y-sjpR8sg=rx-x_xe"
        "sisXnSo{rPleg>=Rw$Hk>wum~w7KtaTQSfYgUsjS$JeZ=N4{O0;xyi{(73xvOIQRad(1ogaZF6wm?Mq*Yx9&#+xlRt18W@e4^T@3"
        "1QY-O00;m803iUT(M`e70RRAL0{{RV0001Ua$_%IX=ZO@VQgn*ZEayMWMOn+FJW_ZE^v8uQ_GIqFbupK@IMef?E;CnmtIn|Xp4Pd"
        "FKLnFRxAQWqHVRZBuI3z*{?4ZKiWfT8-`2`XUGv<Rn`5w_Zw&kd58q`$`KVC@4zJRA+ONF;MiC|=#f=N8t`Os25D-HBVAWjRTgE@"
        "MBjr(MrAQ2N?`inBmWI8pp$95LfcZ`W39ZqC@=UO1)mK2c-}h<sq=g-Eh%goZ}pt@IQFsiyd%`-hZud#f@Se}Dd0wE-pGAC#DQ+h"
        ";u3>{|3&W0{xA`wO<90k&TBgIIY<C_2$p^r$1AAo`eg)Y3fx`W%(qxGJ9^|SaKyyN$?J48GNh@!&6So{;G=1c9Kn?Dn<f!2djF`("
        ">t)uyh4rWZbtK31IOM<v4v}arf=m%kXC{c~S~)REo@L=8Yu*H*I?Rp8YAT}6EoRCc37-S+XqlrRFmg)Viz9{9r|h7OYsHaZjWP@!"
        "I&E{rH}Lh_{w|=dN5+R&lQ{24{ZA1?_dWWbq{JGKK}ka|RX*RR4<IZug3dFbo2?d*JZ(;cr|q_0`W_b%O`@xVG9AbTEq_I|g6g>!"
        "gL`reR=gRorMur0lZ^g47$;*70p(LFm0l)e0Ig|VPO2wKnX*4nO9KQH000080000X0OQt%K@9@{0LKXc03iSX0CRF<FJftCZ)0I>"
        "XJu_|VJ~E1bYU-JZ)|K~bZ>GlaCwzg%Wm5+5WE}kKP+^T5=pglG@1fw?1O=#2%4Z5!w_g`<q)Arm84rcUtdxW>*+iM2$00t;mmS("
        "<v7lAcC!G6sSa|IB$)V+6CFPXg0mB7B=Iv!NQI0k--qys2#JW*m*w5lUG%ViSp8W4T!mQ@I*v1)OeZwUh18%;z_Wc%rO04MQxOxk"
        "$1EdZEVz~=)(R@@BmTKD_jkLk4^QF+Ieo?C7v_Y@)sbX5LrXQC#Ed8f<-)ySjF@c}(+QX>!4#s1a;l@qRmf87BZ;(Si)gX{$~D}="
        "%m=#XLZX2tc&tAM^V!U+nCxW<!gjpDv|oG3Hl;zfU#;v$pGg8c>%-ETGU=yEZ^n&pJqW&aQq+=3Q{X~3M9qr5B%uo7H(>`{Nhzf3"
        "+!Jo`agjO%!U3~fpI~QsL&t04k;fuI8RZg{c}C-EvlD{3)m~u@po#KqdDJ3ILeF}(9i@by+&-;mz9xkSlbGS^yM~sc2+ruyB)AAJ"
        ">(;VXY(cefR%I0mjw7Zo*yyR~COxd^t>}~<Z9c%zv=N%fJ!-qsW4hp}(pwT1R!RoQ4$82ajs)97#7HdaQFTmI5;MwO_iE&{e7L$S"
        "y|vh=_B{9svkz%s^ZcMoRj^*FQt=K=IylEY(ey9&%Pe9BdkYyky5{Lq_0=LWR$4?|scqV!We=`9h?Z{%V18-uW{Z{zW!l$BXwj)*"
        "PZ>Zl9mDisYTmUmew_%<AC4X$5`D_?whUr|@4LP)a#lI{G^f>)5%qE=eMKo$#OMpwE!vH@gQ0g8m*OkEZ^kpSL#8^GdclLC+iPMe"
        "{`s9l<A)5T8F|gI^vb#NwJ9Qli-ow@o@Hs%c&6mXU<2wR9UtO;k_YUD$0FNNj`k1q#>lxInKr$dU|Z8>wI<vwg88<KDPt|j{*l}>"
        "a=zGlqsYzT-R?3yWQ}bGXSRhaaEC+X1AN_}p0X;YmF?2`*x^+x1A_i}br^<6R2FOtUu9=(%X+C{kf;xmcYg6ro7KhRL(@4}e=7Gi"
        "RQ~}`O9KQH000080000X06s{^4af!n01*-Z03ZMW0CRF<FJftCZ)0I>XJu_|VJ~E1bYU-JZ*FvDcyumsd7W5WbK^7=e$THE-WPc{"
        "b<+-oB@hM%*co161{j#3oAD^NoM>yyMv}YDF8%L(y7`jLmVr#$*g8k&eCMj8CQ0%eQ+sYMD#bY|x#6l2QfMOzQu7B<u$mODH2lGk"
        "w?dZfTTS$VDPEG@k+8ZZT0D@ZEqSf8BuN$v(R8gc<iPZy7Q6WRMz=CPH_RO3v*L?NwGDLCH7|^4r6z&&UE4#i^7N7!R@6*uKG2yi"
        "MPZgBY2jISqG|zq&zpv27Q14F8Fu}`yH*R+s^hPcb^6en#bWWzAcI20J1#fBsh+0`w*(;0{y&`g>jlB5g9&S*jdE|=FlF*uNWLaQ"
        "8n?z}d9vc%D9vPB(JIiZx@8tb$t#ju|7&X1Ilh!!d`)VE;}$PluUm1`D~YeH0wRQfs^peky!-az&wLvnRu^AvpC?OVC1rK{mMfa3"
        "L5Tu3u5uqv>c7@=E1h1boN1))#eS;f*|TTlm8b##y%&&Kmpl?`ZPvr$=$XfhxE4M0n7)>AE%@d!aVjEE&23KPY6w{#Qqd#PA%N>M"
        "L0viyp>@xlBmKK+@3<82czAv8zZ&+CJBoI}A{+AQ>Wj}j&Es=;2kOh!xG&U1BwbxE7tVL4?`m#?hWtd9hR~rvewOu-bp5VIfjM}W"
        "J6<6VT%@o<SW)oK@bXKtZ|l-!n!d!};y_S#MDlyC@U-t6ILz0aGqoZpTsqpaoMB&1(Wc}paz$E2WNSPK1WJjHrt2ZZiV4lh@2u|m"
        "4@$L)CPLn`T9h#cxsMJi7Ei_qxz|bRw*S?0b%ZlQ0n?DwPEqPv3MNZ|qC=`|$nDmLlXHnT2&0|}ZXp$N48A2AwaZHyL!Q}n7<I>w"
        "o0>Jdl939P<dSe=1e{_S-LjJc6(jY$nI$5Jwr6-AmF0GOuq^`3SXt6R_DFk1zHgmdeUXuaC`*_HLyf+K=PI$s{sI-HZ2<OMas^!_"
        "JW?FcfQ5ojTP3*82ALgd$ogqe7YjPmfNE%RIROWgnLN^&k)5wY{Z^O*O|m4lsypkD_lZM>M&Tapw?r?`(a?K1*uLMxkX9v+E*x%?"
        "!!syI=MJ^mA!N^|fUG}8@Q~+`07`eXDzZ0DQ|5tWP3hm-rgjfSwf74TCQ1KIo$o)vS)r|XKRF7$<Giy7LWdtZ8jN#JvIZ(9_nfYl"
        "I02<^Xz9F~PC&TQ-JrU&H?0UeZoRa1kosb_f1A#Y7^)#-okd5-)LIS#u9mLC&&iu2O?}f3g%-R&YIv%`Wjhpi=oBHbYypFWa(4B4"
        "QrND#UXbLqOz`EE`)qOebNA!ho17r=Rgbv?{X!PPMJ-CUf}IGb)b5)?ums0p<=G{VlA$xQJ<s8bJrJRl^bNJYjwidfmTDTXofDjr"
        "OXtYh5_qrtDahm<3U%mLR*EpZS&{$?!rq~YjutMb!fhZGjsl-}V;E*=J?jW{>^XK;0Mja4&2rW%So)5e4<@<S%wp5@Bvq}uA;#;u"
        "wY#@7x+p7<8{cI1^EP5XSBNmRJ{qmWc{h!s7Yepjc4Hk%KSrC$p1*l(Uu?*S04An>KznZCpINmC&MaSN*B`6~ytdr3Yw*xjc75Xj"
        "e}0445Uk6STuVP@49d@j;dZr+J=?~|AGu*+ectk$NuYh^#ojO*)aY}cf^J`c5!mgKNhb#+W;}X<k<4<UorUf)ozddq9OnH^nDc(L"
        "a51Yn*pI`w;VLB#ROspj4h|ajE(y-ZTz9+>tPTT-PyCd8Jsr57nm4SvoIeztmwISSyS~JACG;gff=r82-wAlhF06(k^}Q@Kc6eq~"
        "IwY|x32(J)WWJT18<c0UJ@0}j^uIe{Y5{WK>gl5{S|1bR6yO-g0n#b>C?X}B&1{~XY2GQ$?xx2{-1e`BNrNdElT<jXDq%hn$WrVv"
        "egJ8>HfFeEUf+Bc(<!<(=}mR<5<@_C;17OJm=(hnWk|pG1G1!RG0z>7HJ;qqo%_VvHP<)M-2DD3S&lpG;GOrHm)P`*)tXOeW7t5P"
        "Oxq&|ZkzoOZ!rDOI+;kE9`z77JNf6x;<3G+26Ms_k}Ge~zG4K{G36$F!;J6302c4;4=`CxxBem?(~ta(sWJ>+p#Ie2rgu_?F@3gP"
        "-eb7#Axm3n|Kl)D-ZmZRQ}?|2MYUr`T&7(+pjBvaRh$C-xnKR)+r)O-@7}vvXENLPdDv-vOvxu3#}yWyyiGlQ%?RmyQVoMR``?tT"
        "?3+$UT}y(@=@rkJE`-?ltc9a7t26jZZ)mc#L-g7fPsl;9YF<nK08mQ<1QY-O00;m803iUPsC0nx8vp>VVE_Oj0001Ua$_%IX=ZO@"
        "VQgn*ZEayMWMOn+FJpCbVRUJ4ZZ2?nwOo65+cvWQzvuKjV3~WjR3g)o6Q_=|YEGJ{UEfXZtnF^w>v&m8f^60lsqzuq&CPee^8iQ?"
        "1Z5|?t3E6dz+f;K41O~h(DS^P)gmsVSt7-<N|LR4manTqNO%~h7b04$;-ZMN^tnjodzlM)Bj;5q3sGcMK9^#iEo9(%-j`o|`9-{1"
        "XL%_uqvA4&XZ7zmtH1tMWNG~)m-W{&yOL@A2RvBj*$RjziJX_vZ6Q>RS6P*oG9QYCTt-z=F5-EqYOkXb`_nbw!cXLe%4YF0n@7oo"
        "T&<#D5tUJpEtg50%CK0Lx`i5Sm`fP_VkiodN4SJ%GGFI$TJ};~-<FqI8vb5INnGA)g8r9Z2>6`*a(Mdk=kWEx%kK}5PEHSAorG_X"
        "-W<I8Wys}o98*}_re$;!UPpNf9TuuguGZzPRTdT%be!L+Dsft@K}w<Gp^WkiSwh*Mm)l&fvm!3Dyyg_2gz-YAC9scDJ3Ko6;pLly"
        "|9yFSaCE5H6%*m9PKq+G=4F-3aGlFS=I@cM_d7KGCr3XXznX@})9+58#Rq26L*cHY`PH)Y_Vq(5h!^m{ORIIBY)_%mTd4HisW&vL"
        "<nn!t%7+z4Prsf$neXlGET24)^Jx6+@y@fQTr76J-hJ|TZ#jQFA3Yr{A1_-3aMXn?!z7-|w2(+=6_>ioNA=g6XRDRwQi;>a>@2=O"
        "ecv|g_cFYk?>=41$H3R)(fsKmik^(0%y(wv@y;B0pY2Jx813!Kr=P=cDsM{UBzLm8isyM&WXp2<5bC|cdas#oFSE~fp2(fY%V=jA"
        "$(_Z{Vz&EqZ~l0>w;b<{#=Enp<EPQ?<FCiNpUCXTFTZ#@J^ta~^mO_<e0B7C8iGDe4^O71h8S(U2!whmw|0V2{b*J!wsyAWNmM}>"
        "=EKMK!?>8&-{<qu*4Us@rJ74-3;zLALiQ^p^HccveXy?@c;3N9n&oKWRx4N^#de(HIuY0LqP%<#E2oU&6m3N*v0OyyLd=qEekEqG"
        "t`|`r%YrQ<Emk77Xoig+ZT5ocAqeN-@cW^&?DY>XkN-A3b``!l`r(J^;i;?W@Qt1b{!`|+`%N`qzsgflR_jFi&13vnKU+lMOPEG0"
        "c~%DH?OL99tHw*gWniDghlZQ!Pj3zmr^ANS!QpG<ym|0?iVr?^Ny?;<Koo2bSeth7p#5{1<)XxnYoZr2l{rk2p5r(RPKeII5zjXm"
        "#b|*$f<>wsN172yHG@qp)j;0N<r-{$-Q_8MndW&0yVU{dOoUsRdU8T0oqi0u^Wl3x9=?5fd@_Cc?VD-H7S$V)mJTeXEr6j2O>s+a"
        "6`k%D=65u`L+}!6Cz4Bttml(GO_*DP{&wha`2%rsi8~4|mn>Z;5rBhb9$nDd5mC-^lxbF7Tmmr>6|jAjVkXo1C2VR}A_I7lMC(AU"
        "m!(K#%0ev$;xFRyPFGs6&HaIEy7ULF(KpYVCMW`hOoHMv+Sz^Td+$bG@LLw^Zq8z{FJkeChkiZ?fHxSRKNtj;@@5fV$O3vg+aK?q"
        "55@SYp)ja-F)B$Ofxvx}^DU+ZRL81;k~<i$q7Z1Ek9g#$D>se#nchTm5a*?wUln9rh~k!-CGkJed>$cM$=0Y{QIgz>8SF@4V`jJF"
        "<nM3dQU=%$lKZxn^6!;I!z93N3IuewLJ<taPjLxfx7Jyf0N9IifLe=cE#^^5wIe3Aiqm*it+r5>Shk2^8<65sR(V{M@%*_c<SGI@"
        "F^Ac>&SS)iIXnU3WK~%J))2)!Tg#vxsAdusF${rx8HUEh6*5^4#X~R@mjF2~vt%(D1$)D$?^<yuhgm8SzGa|yfKUJksz<oS!z@j1"
        "C*MV2Ow>H=w>t^y4nWK6?^c~|uec;S51UpG#2a}L&2L4PZY|>CO5iqf1I-p0t%#@u3DL3vOALYpgoa?W2*fcky@gDh#Bd}sFQ9&k"
        "@Cz)<dkHfiz0YFMkV~|>0Fw(w&s&5rc|QuQfgsusb!RXJ<i#;KR6H!IC1{b6q@}Q{eY6S;@pQ`K6g01--OzyZJ}lUQRdX?eIyi^M"
        "LSD9JSi}V}h?Vv&6V4d7_#Lphe5}<>f2i?<+y&nxIOtXj)?mNVGFV+L;@syS#pE=vK(*uzsC}4S(a-K4Qp?Tatvy1;VgR$6!qoZH"
        "DJWo;ya)3WL`4Xz%O8Ld@m{RJVodT3b_ICm>J>lj&Av6B41mxr_Q|~m%A<4a`ugLYOy5jjor?ESQc3aM@zD=r1?UIipZMvo)8nbQ"
        "lDCtuV0&DF8}tY5c3*jJdV^po%Q*mS={wj~d#M)3(Hi$|vTn3t!lof4e(4#T9@0F?3MYO1<WUUP>>oH5BD!hd_3jkXstsF148b*C"
        "MKG9snB`d&Cwc)w&A!nK^2}%qPUxg&=T8&iyF9Vs;NLp(e_?!%E(m;4Bq2s1LzV%w)Wbcrz%Sa`czvV5rEO`Y*1cCpKOUa?4+pG%"
        "9;^iF%2cPb(K+CRaSv^`01ndH@HR+<Mi{t;$oQiDeDf-{h&&Ldmte5q-$=l201{CGgC2(}#1a9Y*Frg00)#rR5Wy~Bb)Y>9oZV}I"
        "74tk^mkyowULPO56{m<R#ld%C`t!lb=?Q_k;<<2^>FMezruls?UL8+go=#OMk@IytL7Yy1J{50|55Rx<Mf`30%TQe8(W(&NzB&4K"
        "U^OLjehq3JZChmR=9*a_dk_bQr_=AjV`?=S8q}y-T|5|wpAJs{di3L|I6nI6;PrEB`3^RU-1l0XfP@Z@PPLS@q4dE-cE`jPmBwSM"
        "+ui2bI{-)e_*5Jni{t6rH!ojJaok6ZZT`p0H$Q@>?OT?4_+ntX@%qzaaE2XGG-xeaEPRG%T9V3DG1j2ogeh=j?!nDePn-@-Q<Ifu"
        ";_y<8N(`$IQWP0t3!32q(0U(kJFm@+1NM;1Nfg?^!(DyfF?BY+GJvN^3=Bs<faxe&Gvf!}!*r<$XAGui`vuLfXqHklI|EC{W=TeN"
        "i{>W4c$V+|1#PSMcTYSd2aKf%_`4<5yB0)`W|Dd6Fxu!2@^c4ma4l{(uLV;wwfx(CI+ULcLL({K&B&IZ>(9F7SPS|q>WgdgXleqm"
        ";|k|GXT{M@Yu1^rbPn@m=_7s|T=oG?4Z6>;{&|3F*WjR8)U@TRW&Iym%=c3KJiERZLe=P?w)2Dzcs(#Ot@)U|XfFy&>ixJ>dpN)o"
        "A`Hazz#SQgwqxSZ0o~Tng$rpYL0c|ddxKOsy79rgf~lt9xw7kW5!PrcmrDu$$sEJ6^bnpE@hVOt@WpT8;W!w5{8ZbzB$lYIo2RxU"
        "Y72>!t!ewWS?2`&E=8sjlOrb3_eBQO(UJeDN{DMKTP_iV)sdaikgQ@eKroGaqmc{H&M&L<$_x$^XNuasCi@86I%234J;UGwdioRL"
        "4ON)Rd!fxT5w}ex%|~`!+}pXcm(Q?`vTN(fOGzOsfc9=`k6L{9)$s=6`{Sb@-->U4srz+V<XVJ|vL5Hd1R)srz*z%v^f&~pABKT_"
        "G~s6(S2VW@sx{82L#(wWGA*;Q9!x&5sAZbTIPgZ~+mmZY8|$$amr)J|pdPZ~a0QbWaxm7n)$;+x`sl#7o@AR)LC&5sNk&X55GOHM"
        "7n@*}LYeqx(gF=7`a0Dbym*d5p&&Sz^cxTZM+fk4yx<)tuqnQ5hZ<>j6WI)-ULZ#dsfcxymvNM+A@3{yiMxtzM+w%vW4^jrty@RO"
        "z<SJ)yFp8u+lE-cM{P$Nt$B8tk__fLtz>6FcN;PMXqfj&Hi0(=vI7j;*a&F7<L$K8LdS}<T=Q;)3=}LUzF<dW2wJ5b5K6&x0)}FH"
        "TkH+oDABP}+g&&@l_gI@%Y~uH;?GRS#%cAY3YA-Jx0z`*-H(V8QxIFNn%!`rLG+u6r6-Q@X++R7l^Q3|$${3_@Haj==rg)41W`|L"
        "Pu&)>Xnr`V8<?Yo85-cpu`3*N3vzynqr?aEHiFw29J9mB`bexkVh>>t@$=}K&LDW>*8HvWY_%?jI(W}|169e6dtB;)#@3ITirU>*"
        "&?+wCv?wDGy03}Ck{Dq0$0}|}a24}ZtP|d920>{&m_kB_pM}+ZE(3=Bxp&NmHe~%T;`cHYoDhM7MU8$pgN3hoTqe;)F@e?xckq#="
        "I<G=?yhH^<X-V*oqMW+^_19nh7yDfA|JJvIcg4eZsUJLiF?g3k8T|Y&SKlYiSePp43C<-7k1Gdcg?;P*;K!dPV&|Sxi&nGvqROfw"
        "jMH^hhD2WvVJ4RuIXv_<xX80=?T;x)|1MP+DXU6o$K%9ZMoSsSVvqJkcetLZTJ9064%Cp1Vc`Y}cnko6%IjK4V4_&#xe>)5lO(&A"
        "iy`c?Hym3`qLcze+cci~%r<DPMUo-I49%GB5!_3j%y(O$_=%P!S?q>9mstj0S7S16+2fMIU4@q=YXUeNJX^4uaxfI<uBxb$)y&u*"
        "bMK#bEekxFK%9f+%M4|U>2n6%cL7akL)0^W7!{4D&GJ$Y2lr(6r<zwZ-#Mr0DIB9V#S$ikG9@k>y&xWOjR9|FC#rMGz*uVoszTW+"
        "4-haK(az(GI0Z;%q?&<*KwfCP0&z$P4e>5!%up?X`6Nooy8D?28(+AL6Ur_svZNvmx^B4)pkUW`)<HUmqwwes>ha>Wi5L|~!_0C+"
        "9=I_&Cro9MLAezK;qJ@;l4oR1ftQnJ*XbuDS5cKlupuU@(CBYdn9wqwhnRO4;%QS@rRpSc5f)M|tQ;_yCT!SxTCHJlGFsu*=V@=X"
        "$tJ3^M(A83Tt_h?T~>rb>7n{H)TXj^&ZM3`)TNj;1r&%G^$Z*X7`{5v?^X+oq=Kd4Yl+e-ltrsP<}j#pttJ)V%P))!FFavKLr8Ao"
        "b9Q#No1h9MhN>(Gg7lckGZ$}p;p?2w3-s%nL@xkWa?NMkXXje0(cNi1LQ>|@c%9*CJ%<TPNx<eq8W{4%(T#QjiD3uKgo$Un&Wwri"
        "Q!8J>5?agbs0lz$T6v@X%}~!5$4zh0u(W1XagK$o8AFa#pa;yFVnIH(Dk}BL$Sl2cv*DH@v<xwMxdgfI(|I8m+%rIAW!yrgO$oWa"
        "<(#VpL+5zFzD%nXC87B^X4}D2aR?78dlZBiTL6{2T*T2uiZ1HBF!ZRNYUg>sp~@2%pk+QM3e*~MpG?dgQ;xSm3um|qWz$Na?r|&|"
        "W`?SpiJ2U1^bP>cnUc$VaHV?%g*3X&uFpKwX74<Z>7uxf%S+$G4WjQLdJ}@Bb>TW3R+gIbBUQY(HuC3s?+g%i<*pNY%y=^?DBbVN"
        "pVdK$9%;KL5*xMIoHV<SL!|c>4{difWpnM}(lfhso~na|K&9^Kt!-<xBpaLKZdMy2#7Ao|)d0KChR@HvAx{l&edj%`VK#2sc*p#>"
        "^Ii@0wPemd7W8*-fogxP!BJte-Mh4p{A3s-)*pyJ*Tx#NW}HW6TC{j80OPj4w~fee3p^ytdh=#W0wuskGKFsb6@_UKfiXgr0@{~c"
        "LLe4%OKAQWQhDcUItj1AboQr~V7e!JE(=$zsGrvq!-KQi(kXXEoOqi|TD)xN#A(!>7dkeiC*g;4brJmH;+ibI53GeB&pxnO|9HOt"
        "0h#&=&DK}vAA5rL_~{uPcjLIK;@kI99K-6Nsb(4mZ9zj7hp-NyEMcqGpE8M?rD&)lSu))L8ZY(zjH7Ai=wh~?H}NuuS=b<3gmrZK"
        "-bvMVlkqK|qO}SJL)+zOtqNCs)@ehCDLR+X!1LZq8elRQdW4texg_lHB(8?w8UEj8cVW+;#VLW6*4UI+4}Z)tIp7W^El&)1Tq_gP"
        "?Li1o{r)fjX*UC{M~BZ_n&9Zm;ENJxc(>5%1bdCIfY?xM;MkTUCA{=eassyg!z=vs#FFu&ur+vu1|UD<hv#Q_`RKeSPHeOeuO!8a"
        "*$cS4x^YX%*=t?CvG1-o?IdPb+ep*IJV4L{IrnWAf(?n6s7vY|R+aJ-SUEbB!cKcX!9LX9--2Lt%gGNHa-Pt66=FBKbC6YKsCnr-"
        "DxmRvW)$K-IG_vYlXE3EGLF^icnt-N?{I+7_^1Y(6&=PGs<>c?hnGOV)%cE%j2MQKex7lts|5R$ZXDvKK#sG)JOW1t^f3u_WXKH4"
        "IGeDQZZ@87@XOk$HqbIQ4m56Bc=&x&GZl&nF%(1kM}eBgU(1zKvaHfMr;~?t^32O_6J^F=Ht<3F)j>M<VUPm4(E3ahh5=f{H-QLL"
        "n@+!XEQ_-~RdrmnbMF^to<4jE(ZqieWTED2+t5heHtg^mC-{C4MTggdXuSqs9y@R%3681g+RngAUGZ~p?Xc#=bp=D+V`z}Wrke|}"
        "Knn`w<Hk*uHjGHe5^1OBmdd+@%^%O%`$no8YQg;ryZ_yL_b1yO9toKn{PLe&l6Px90$~+VCjY%1qx)~MI~tA9bdMa<sT=l5xFbh|"
        "1Qj1T&^0LjH5l?f+8(2&nVUyK!DJ0?@UE3X-4W`vYFk~|q38B#x;nVgYhu_s!chyp9wqf;o;P%7F64_ah|}(PiGoSMV%e0J-KAa*"
        "q;mYVeHhfj`879ut3}vn1RSAXCsiS?ja!uJj;8pA0`S5gKW*nU>ARecYm|fM?9O!V$KqYg!hIK%`q?usU>eyZbiGgJWed+4XE|g;"
        "LUnV6;yxJ4EqwKtk{=M)OvYd1J2>70sEcSXg)YafBb9YV{tEydz7kLS{uZ_sHF-MT1(c(&27_lE<8A=Eig9J0qqigRmsW??9iOg`"
        "7N@kihh`%kx#7EenB9Zfqs?tyHO#@Fl?n|T+`^B|kka*K4Dx0wXYo14`eMrlG=&rWy?$h895q#9qev~LShIpIpK<&{>3_7BT89g2"
        "3<f{oYf)fTp|AbghwgwjlQnAf2%91JQ#;K%YX<-c&su2FB=SJKVkJOL$M?lhJ{(wxJS!-ccvWnxqaU%Vic((^E;F$1mog7r{IFg%"
        "!_a`*$$b5;5~`2)Z*Tzkm-*+E99Dd6<qZKW4GdC7bu71Umo9Z7&W~#&;F(oQgz7+FvGy6Xv_k|vDQtCh3=nd$v3S}SXi=#LpR6Pe"
        "xtiB5x>EP?o4T0CxH*l6lRd*#bS1&G7ns2e+ct_@uSVcR=H{Ip7gGUUk}7=DalregNwk_RBF^x2AyO1H9W@c<+(xma%sNFF)_ps}"
        "f<Xo_wu)~2G0L$5ahR4uJZxDd9btoK#_=3^XvS~qRGoXO!5bo6NX$TuR9Xj8Qc{PnC~n0R9pdPxoLxv`fEq0Z`L!9yfwPMMyiPU8"
        "MKj50ahYqCdIr60oMk>2Z8nb91k-shK4^}6eP8>n+z;R!--Av(+ExapVk&9%<Z<(<9&EF!OL9!`TMLnx$KlxB(3Hh{<Cleey=Kd&"
        "BM`PUp|+cnQhW878)dbfNzW+Sw^xu|m*v9Mlx(aTD5w2;^6gTJtL)P@?Pyr5sybLexM@5!T6}EQf`$iljXlaSRD@H(S6kJDs7pcX"
        "TPmkSs7P;Jh=$;2u#zi0;8{g$O!ejFcv;08V13>P_RXoXkF)u`2ee-X@p>L)Ye{dT0w=@ooa`(23#z_EmJS#Y*g2!rDgbF+`J=%-"
        "dc#AOZeOrwV26uJ7!|h~;!CYo3P6Q3j6loom1paXA>1_RHguh5-NrIm@J({n4jm$D0LD8gvv7S&_-6A3HDrMr@qr!^+T0mcZj;yg"
        "fwBqJ8&XW&;71@x>(wg*Lp|sX4Ah9^?0fl))#4H?T_P=mz}IT<x*Mi!@FuXYs?xiB{PAe#30P<RgW2L<NA}E=4ukVmnwHT}6^xg@"
        "G$Z9)YH;#M9dt7W$^47o8lwYN3u*p(*sc>MfY8gy&gjWryLz`^78R0ST*oC(o{n^hEB7f{U`hr>x>Q$=^ryEx2=yMADvf`yD9Wib"
        "TbH?%e_$FT#zr~IYg@Ac8#KP|r$Ou8SI&WbT*epLJWPLnJ3T(YmsN!)Z>O(L`dX3UgHDZ?)tT?=({EWakBwRU+T=7oC_8oqUD>)Q"
        "FF9{!sQ&v5n3Ph^WamcP(tGgWfjBTyJySk$HF5;MI{)|%o@2%DK;-+PbtbDDu-w~+w5F*k<qG|PhYz7Pg$;-!3b@iYBOK7Rpz64>"
        "Ux2RLOwgkvh+@y>*~k@SY_9|J(_RyfHcH9IkH@30clrrzs%Bqx)_VIn66g;23Fr{NT1bz@&AOA*jf@t+S<!w?oc?&K3k3miLId4v"
        "yy)NwYC>Uj#e~^T^R>?9GQOea3QJP43bN7TEiracu2F=mk*h&%!P@NT#~WCxt88Y=iiX8YhZ(%%s#KJ~u99LThA#kh3Fw!CsU<hW"
        "K0?gfh@lR@^ot|HuCrU5+cQ0l;dJL)My;*Kt^)wO;S+9Hb9+|{s*SFU{dm~Wu=rIrQ+tBcup1?H-KevB{-(V&x-HO2;6^QN#g?t*"
        "v#DiAxv+J6&$O1EGZIf5#^;B~O&4?DadvXA5V*Roh4H>+l3i~+pHj0#m#_YipC9n<2|myWm>u0yt+0;8v+-=^-e)4#EPS|Is)q2I"
        "Gn9{Qklu>t`!T(bZ$l)%j^cmt6xPGOsTp)vpDVEm(?9Dzf3GkA+kX2Q`|>D53(+nKVFUb7#ju-N%y0yx36tEMQJeE7$5A!4X0>KR"
        "*LQ9!Ni>tmzUt(RLMJ#V{!MlkgB^9C!(pC~4Wcrvt0`B1y*mQOeiZDIo4<axOTV7!pU?Q`Gx`bk34V@k!}+lV&D|#q#uS+$5Hq6R"
        "H)R_S1zJZ2wKR1UfUFACtqa=MJD{*=zn_;1noYcA7HyB++PGVW(5~y)S=*-1;oMWhqFUoSoC~>7TNkdNYK74<Yt`!$Ycq0Y%g{JZ"
        ";Pi_S!!ykh0$<h|QVcn-FHz=MR_gcgag9M!u=$mH^iCZeYO<DF9eU!ok?EKdg-}e}uU-R4Tx<)Un>-R;fbS|#%#2SJ%P23En+leh"
        "rx3iAiC2}&tvwIl;dPC2pJ2wC+l$rQ>ChI7$6<Z>RU>wgOGUYQQq~@bV}3EQaYTjJ5Xb;<4lmyLLhHGR5_IeU4ar%ST>+4ZU^er>"
        "O&Ezd&4>Q5SF&NRvTk6tlXy3@vo2<tIe>R{r!IvV7Xa?+=-w?Hdn$fZAc8In`lv#|qFSvBzHb2w3}2ld74tY&%Byl4%?l)Cuu0D_"
        "m5k2~tmq2YUPWoV<nW4V43{|bCBLt{ajZl27CBmz;rnsx!WZ=kEta+w<E5(7fbz0kL~$y>;$0)h8Y<2<MBQ2INymFM{JgtldKYxp"
        "zI|-V=*%<D2Ktg%bY(@(S4P3p&&eiu1D@Rw;lF_qjmg&ahl1Z3KG|{3eb;<2wRSK{6Z`Uu{|8V@0|XQR000O8001EXe)w6WHV6O!"
        "zZ?JnBme*ab8=%ZVrgb?V_|G(Wo>O?FJxhKVJ~E9VP|e{b98BAb1ras?O986+cprsXD0svr4Jz!iFVvhr&a4_I%(SJr89kAWDWz7"
        "kcFBASOB!;xc=|m1;D2$S?=Rj&LkGe-NimX?1DiM{KB?oE=6%i6r(v)B<J)>NNsqgzaTWHm0>xllr%Dv#f%7c14MGY<{762Suv4q"
        "N~*4j&ZJ^G4uatMv**uvSxIF;Cf+=Mma{d;U=>v?g+KBSni{IFb(mB5%gl@zdCdg>z|@SqoRN}l(@Y8j%BA*=zWC(q^G~D2^JfIE"
        "s$La5n}JAvLk%MrWSPw1pL@HOir^xI3*zN3gX+#~qzDo(!AhaZXv^C@1H*tEpWZ)oPf=n;&1OcdQo^8U4F<A2#FdtTe;y<;H>}iQ"
        "6!jLuYva6oLDs>gI5{~X-|~Vj$UTaExD>&V+N-g|w+S%mx(o{@!YXpQ;QChmh`b{B!ikTDQY=LbK4c;fJxOaq)FWQkMFFb>LHxVq"
        "u!@yDG@px{Z87{TFBmS*L|rlk-uE7GZ*-81E!VPE8OCVjsiot9(14LalEjcoGG}2>oAvy&U}VdRffgvmiCZp6!>$deo38E*Lm@dk"
        "AVje>Q?N`=&Xg3=2?x&|C))CZ*d}r#<9wK$6(d%@wgFf=5eqNMF|-;-3Hj)vGhpOnYaHd8k>6-hvv*2K6$U?n!+-p`34pGsASKr&"
        "HQ6Q}>(7NYs?H3T0;~vk#e+1XI8;BOBF9)VYKYk|!nQOsV1aPVIn$ZqD*%lZE9C7ANKLH-7gxl>42SAMKgK&rkJqc#^G;5MGJ|AD"
        "80{&o0PE@}_D9XM0qOgM!E>m=o#j2=ke-)!;xyqGj8Y3;O%8jNUNu5zfTVe<D~6!iAlfggW>K^&@v8~=8#&-9;m>4Q6%0_)*Dl*;"
        "@I%Aj3P6;b*6>xGUoo?r8EYb>>6{WnqVOtwk`An_%$-+YXMuvPO$s^0Z!Ajit4;3~owQ4iIEp2+u(guJ8cZ<-7fgrO3{q3%-o2yx"
        "N7)VUbeWOG(b7)HJMY37LFzdz_?3Xf2Zy%B^aa6<4r)f<LPBs2g@*74Z)kzlAR70_L*a<IIR_mPt$<2Vam7MN47(4Y52Be<9c_ws"
        "_8T9$zHhN^TajOqR?Y+`UK!By+-<K&x17Hw^HxC8CHFM@bhUIrJXkf&(od?J=KAvD(E@6&eYRWsTRdphnI>k1p2FkK)P8z@OM!Zq"
        "_<)R~_`%zQ!kLJ*OOzvm-5CDLiCnfJ|3A<N13f~nr$_oY0UnRXm2EM)Ct%MSD`@ktCxz|_;=jMY;q}#2=m1_mKHsI^7rQ;CoRFW5"
        "tcdSU1cRAF!{#PykyB`3V5)I5%<pws5oxEHk{g_IEab<N&S`526hA*ZJ3C$_o+I3^xo`g69I{y4+iv&;{{JF*fP3Gnyb!cx@MP4S"
        "xQQpG74Ak8k-DGOC&0E2^-g&nEPA@k_Rzat5l91eLyM}Y^{JcJg2Nmz)!lH7?G><|pPhd)Kl^lk{_$QeC5v{Iy+pWkrX}PXQZ@}H"
        "tYrJp1AM(+7LD%i_A&T01mc~<KFpC;62fwa`uR~g{f?C-O<gaQ7HrEid8KHz;aQ6LmEOW6C2w_l^LnrT&2*hMV>igRU@*b(Fnlzg"
        "0NsH6-{;NW4OQEC0bl$3XfxetF+L91J#xsD)Ea@ypE@WAz`uXwAa8A>2C{qTV3)?WR^G;t5n*5-tYP?XRtsk2{vmQN{e94Hk$c;d"
        "m1U<F9Bx1)%$Ew<d()qgY}Yq|*^fi@<m53zy4@E4Eo$`^R(J=21`kxVv9P*^=?}105)H-*Kw|}IK`}1|6Tko;p;AL@R>=&-vzs7U"
        "y4-?1OnO|0@NgohpOVi#45TIfAjMo~9MkeN%l75+6bfe0;4_t0T)<3_Vk#<`F|FYS!#^qJF<2bdnZ*-7-r=HC3_~mo-ZsOa+Z(sW"
        "&f<9FU|5M!qH6<@z_+@tV7jq~e{9XSF-+FB+UyjM1ehmW)Hb9ua?4?3tBpSyz;PdGJny7x9qR4}+X<irk!W*`U@I)hc-ZLR2A-!~"
        "x=k*AEXMYLr=Na@aP^<&-;-lJ08y--D}~(-cmU6y9yRDO6ctjHFxq|qe_J5Icr>u+k;x9Q{#?K#qCaL+EXHZu9y*08c>n%=_-5gn"
        "pg*g-Kdr1cm-^+U2;-M;qDui3fPfCEfca-H4-d&k2VZ_|`#46EtMK*EBm_!#q={~fmDkY1H9kSS?yKIP{IPRwJYdhZ)85H=i1?||"
        "S1jkYxOC$Y55M<nIS6{<_6U1$jJyvKp64N@z3un7yAb<Ljpq!w?Q}ZWa|PW(>Ep}xbnO<OFyIL)^tf9^*cR-;Zu){e>I28Ix`UTf"
        "<dDtdZE@P1)&&}<t+)tgiAIxycKs@@g}fEhR*~<&=?sH0qAuJAgHGV6)<<<dHfU6e2xS`2!gndrLTkvF#h$jNwww;J4Y}NWIILnz"
        "wj2vcB64xDLkm<L#w1Zsr_o1}F1GG@GAPHMvv7k~JnJ{k&NADjIrDnUba35%a7t}aPM~%`VVEmXB)sejtWRO(+xPfk1+43PwU?zn"
        "<OALwX^Njejut{rd)nA!#b5S3fczlEz=V=(4O?j1OLD)C9rKrM<~*XTe`cF&`>X%5;4t^nMtJapB+UndJ7MT{uHk(FBeTKn9{4x1"
        "$Y*5O?zN4>DKy%esZ-dhY#-62kd;sH15(%Lx!CKN{0gSm6@V|Jr2|>zzHi}Hq#;6c`_LX?gCg4Ej5h%??=MhG0|XQR000O8001EX"
        "2z^x;ZWaIlz(@c9A^-pYb8=%ZVrgb?V_|G(Wo>O?FJxhKVJ~KNY;0d}Z*OcaaCyZ$YmehLlHUiI|De;}VePE#w%vBW0)I#bJ-9(0"
        "o1IB^7w{PZEm5|nWl0gGex$em`&AW5iIiwhCyT=gm}!fTB3bV$mVDp;rhZJ)gq2TTl8LIU!z#%$ugLRs<nhNMO~Rz|;D4S8?um+3"
        "weVOLd0U=^`vWWQykBcpvaCunKJk73?VGo6l0%V~mACsXDVmqhBF~za1FQDUOU24KNqO^LmuZ@8Cnf*2=AwH0CNA@X7v^coX>Y<)"
        "<KE?UR`F5|C;%i()15!j3aW2k?{-PHi&@Bnecg=N$M_aR`9qfGENX^tlQ<7qy5olfn?$T)ldvus;YH0HynW-r^Xad@e0+Bse7t@C"
        "X(U_T<q`kuJug#Me8?jHsG4a$4+@r)JPJf!mmwFb50DHX1H!m$L2Lw3vV+sBmXfn5U}c#<1n^JK6hNIS2=goh%mCbyhj|(4-YUQ2"
        "Sx}TYf`hi<(bMH|oC5U%5m)+vK<F)OA_lDSvM7_RIy$>}s`hyn02jb+o;ny3HH<5h-*~{HLsH=`A6S_ISYx-xcUJNPj}i;@td8K&"
        "gauD`a}DeSpduGZm6y#138j(}4YVQ;l86H<S5L?c-`@WC?!SNd=Hqw4`w!pWe)3k{vpki*uMaHa_biQ*NB>42mb$jdE(9hxYlajo"
        "yz@uKh-<c(U0%%N%X4-WZs+IQ?ec0i+g_eWOUAadiv^Fv`26z9nk2D&Rsm0pr}&KXn10MA1uH>PDm{Ft{`w)y5<qxTb#|E5MVX$x"
        "W7&u57JJQQ&co?te15f^#n<Oo%lLdgTf`w}*NbRzaj^(5ucLSxonJ6(5@Psw?X2YDSKO5Ux4J9=-)A3hKaKz9k6_PJezlx#XW{iU"
        "n!(HVGCJSR<MZ&Eu?rqfufusbUBnE>;<AEQ6Vxm|!hSJH_RGibO^P9Le`T<!zptu7+?<{5k_x1M66S|9B2!w6GdT*lDdhnAO#KyS"
        "T1A}Ar}M>ldNH1#tFhy@g*GzT{pL3WD03l@84$j?bjcnja`{^D5(KycB+yRSf&G?eV-Y5xUin!V;?%RnG`zTq=G)7V&Dj-a{CXK("
        "PiHK=n(_;_Tuj;3f-TRZGopmGY(VS}Dt7Wn8+$><e)}fku^00y+y{*w4^U5tL7C^($opy}^~dTxO9da^C^ZCqfK$}1G;Y{7PY!oc"
        "QVxLPa1^onxvV+bk3>{Ken+orFzRW6gn3cQE-$@1{xtFih)|Tm`$T}ek4Oz>?7)X$nY@lHPJpuyVxTwH0A2&AjoxD0hfhb_cY39k"
        "g$?wZ^*$V3q%x|fg0D0;`p!}ENUP>-{V^d6_o(#C6(RWym?d_>l2oDpJH5m9b=W~ISX3K_C>%#n2*^`AfLW80Y|Owv1A&_j*W=ly"
        "9i7Bp1_N=@geYwTqup{;w&+}}<hTdm`ask%=*<B&$lu;yJRh}1m4KiOZ5XyYFoD;SS7pNQ`2c>8+BJqp)<)%@g6XFPhesHuAb&wp"
        "tE$UP_O`;Zt&=n|ge5`iP7uLzh^mgB&=AMTj$}uGrC~cX_|<;(W1h7V1gAhB09FKM8FaXM=>4nHjD%8K26ST|Ejjf5W7WVia9*+m"
        "ytKcwwC1<qEtCWQn=}OmzUPL&hG+&?r~*H(<lu^Bz`AG{w;ULTduXTya_PkH@{#mOkd>Yd_1~)2aJrm=h63HhE~EJBI=YU+Sv(8R"
        "x1hwQ7gy&zisn<cyqI4@(=rYj<BRi%hqKu-y11HIRz(>Je{sGIujA>(V!<zBHa%ZlvT%7718}}Qk1w~E^Yi6=xeVFmmM;MQWq2`<"
        "mP;OP<MYKlnromh4u%Ju8qU_B<p@HsZ0)dgAH^+GXv_wCCiX$NXW%_WH$HGTz|>?!#sTtoJ;i~=Gv<q$2<{cyBIF!{-PVNuKy~^`"
        "Jhy2c-Zflik5_#GxkUa(K@ZWHcIag!_Q3XcMy6|FViVmcfRKVJ7<An$6Ssy8g7ZthD@+5&2kC)pu2vlv%i~~$yxDYVkq*@I0AUlc"
        "G^8(*tIqULL!9aO9ZI!m(MqCOupNMCN=A5e#1zl!14mb14(sd%Z89RrVS3gue(i&^ekdw`1KXhYEyOKPfW@>%57^&~Jm)Z=MxG_$"
        "@ESum{WXpuw?C4wEw?0<LbEiBA>!&m?`U4c$>$Lr*G38<po<mS8cM)~MbRMkpJ9_PH_r+D`~2cdFlqs`9u!qh@&ZU9SxW04*rH#8"
        "{e|^l3Pczyzpmo(l|S^D@IV%$lp9N<y#`v4&qn@$9HCtr6b|UdQETOVYvCfEfWB!fD2a?7IxK)N%+o5#qjOrp2EECH;4zdi4?Ao|"
        "K8UBhMO;eGQVssbq2Ic=m8g=)MQ{?oG{3tT=YWlN75NVOqQ}#6JFD())^Y|Q52yqjGa<6@fhNkaO*;H9bK4T?%`*|<#pNNX9W?7h"
        "C=22B_Nf9c_HI1g%(F-}ZYE%X%&i@FtXlH_1r%0emt_+uq1kkL(i)%8J&KDhb=>kc;Fp?IRGrp4p_IER8Y+`wNrxSTY)BYb8ScR+"
        "Ft&Q&hb=Dwr0SaNcqQ8gI`pIz0mLXp%2CAx<!%;i5A!93J(kxdD$xE10~mj-N-&GKw}FIejAjC2ph{0l1UZ%HGj+6ZD{-1faYx9Z"
        "a%xxcgGM&QTMTe?hm?p%@^?YA9eBl1+JSTZHe29#G~J)PUkbHTt!8k`5L57S+)f*deO^|8tV2>wydM|@T3(#83K)||U5KE*qXd@b"
        "N{1+=Fd~}fHH!K@Ho;OX^VngV?CQK0uzT<_CjAo$lA|DyiH;x`z<K;3c-q@R>5tdb@iiOAo9D~XV*2u*XwBgt&I3F>0KJF9-zA$C"
        "e`&eAK~g(IA(#Eyl`tu)${H1gY()NpeJ8s;CAcc!Alm5{WK&z7eVyI)?6Hysa`-gNja&#_7J$D7-{cF{$$eZ}P}?i~jXWfSlqjHv"
        "#?jKxDK#>`xLA6~;2_3CQSxvWm$TO@7dr(2f$5zH@-k={R4SdWw&~~!Tj`XnBTy7$H#QiY_04cXZXaTH(6hpA7y~hI23k3TkPLug"
        "SP@p1A4w}wk_3rE+Xuup4J4-YZU9BC(FKfT9co14@PAuJ!zqoRNO^q#`E^zvd61eQa6Mt@Kg--lkAdas4}jJ<f!$HQ#?7@hk|_Z!"
        "u}rD^PL~9$4fhbE*YZnZfS9;>#smF1iT=FNz8Zove1M)YLnhIFG?1h9waTi^)!{=#msl4N1j-*~bZB$9!Pq1(Fhu}N9x3rLXIk+O"
        "&Plj2UiDftny^mn*?f5cgP-qitibVpDC4D(caKyfq0sKwiwP>@tPBol4}xkB_}J%Z1n?wkn-+5h9J1gT2lGeR!Wm_L#t6pCW%TIf"
        "=%Z1h(!7;H+$gU!W``twGVW=<25IpM)KDzJTs(@1;0Nps$$x?gnasL}91jn{1%(wll(sS@wE2<ddudBlI=!sg42nwJ8<djQAjE^c"
        "BuW<=*o+-uM_Abu-}m3wY3k8sMVq+rL|Ag30b<gpuRW2m0<c=iYpa56u`)PGA%Jcv*gbi3B@ZR)^mrsJ9`IJ=5rQBvNBB~5!g+8%"
        "w_@QYo>sS114G8uZQ7Au9eOyr1fI%r`}F?uIEGeN9NLY;3ELFVuBUy)(TFO0KvUdUC;e3AlS?zis54p>m9lJCvmHkqsn}^a)b54U"
        ")S36JyJ(RA;($x&=qYFjb^{I7osw12h(q(D);*OzaR7wF>DW%ERZ#UqQ<GSv^r%1fDN{`h`|88ACWbbmWZHEwW#d?yI{YviU|2sz"
        "i94muiS-c)cbkboz8^Riz=wEUa{3l95=a-8NQ8nzKgt}W?GWHyFhB6ti!dLT{ct0IWH#G@Q1LVpE?6SjfikK~PN}c%z2rdO8&I~w"
        "*mILDp^b04UadlD3mi!Y^7ZPCM>SrK<K-hp7Rc?cy#qutnaw(mL0TWIZ6T7ah;r~BGYBz2h+HyB+e4+X#{Z9v(2^|?$kSw(;18MK"
        "3AB<7)T+UbQM}xT#UDdnRNkOP#mM`PKhZA=E;;^kV6pWAKFdTMX4~+@G{Jc?ppVx3#K8lp=3w4z=>3;B$6H3sXH6Z;>4}49zh`s6"
        "oKim9ZyXGCjt25=_+lwzvIaT<_-wD^p5F;ke5HNSAZzqCQ}lapQpY=?!7`!L9Z7yj<C&OViLL5;+oD<y^ejNF1lkn{&|o|TaF8VJ"
        "a|Ov#0Hn|X6a+UsMDM%>ZaW7Gf!&Lg7|b@6Jm@+Bi6iL>s2sL^F{!I1D2r68H<`H6Z<Yrkn;?8HYF$O~l??+<1$tsNpDwO!aQzO{"
        "E0q6`J&TK}?y>w5)dupOpt}T)sai_+YPE=nAZv`-W`lWA{n8~IDkB3WM8po0KcT|(sNTVOmuXdu$h$8Ts3vL~>bo~g2v2CJ%CkFj"
        "_Z_`8M0(<nBi^5F*T}thU!fU+lat^_2rHY78xuO0B=>M4uc7f4RafrF)j*c8u~|t>#IGs%=}m=#K9w#`F`UYkyNbs+z<NMhr6KIl"
        "bRCUap^b5d#HbP?B$(03N@LaCFJqmR#CSV|Vhn=&`!R!eI$fj?OG^i7b;@@#hBs27f3*0uuxr%!1SyZerrT}WR**y2sr4}s-@xmw"
        "Yb~pd9*v_2du3NiIY728w&uad);`7<r6#Pm{ya6xdNV0)>m2*_6}Tp{^)*qy$)=4AMZ+`~Tsw(m8%D70`;n|ULr2VAdp6KGJ{8+5"
        "$pO>TFQ7991h!?vn0syA)=K#*`pR;%X&qT(2`dtBwM`or!zx|9Wz}Gj>;#QU(9V;=u>hgz?p$Vb%!Y$LO(~jZy6iGQ?>AM4Me&2z"
        "&ONvZz3YJ#IoReW$`9Pr=D;?~GS1rG83w3*C!+$&w$uYKn3`#5*dH&&5yrWsv^$(4e#A&qvIou>F%vW@rt*QS(7)mLFXnO<sl-!O"
        "u}2^~+hw_^l28QgvG3;r6P5Bw4h3RSR~W`6=Rw6vaHlN~1zaX%|1ifSeW0MLNy33+z2z?#GY)8I4{!2}ub2)4k79u|6>hIB>1;8C"
        "(uhmoUu%+f>mX2du@CTgxJRI~9`VAkKGdn^yG6HuORk`R<G5zm1zu<f>F3cRFbb=FCm~PL0dko{6&RgG-!-iZUfx5%6-mD|XwD8e"
        "9F%K`3$Pg3u7cffFseZ+`%>;plzTTN49I}%KLYm@i4LGn8@_5;RImfr2894Y3=e_l<JaCc2VW00yS5xcQ#bnB`|z{0PJrSP^wOT)"
        "CpcF?1SGtBUQ!lVT^EHc^%&(uz1#V>XSa)QErxHkjn+Rug7c0gC?#HBp$P0bCjZPDaHWh!ly)!*XmwP8ZQu&ol2vD+vM#D1IM6$V"
        "RjHWJ&LJLHmc&?8v@#2!C38TWM>a<}@yeJe%VKynBTGptM?zmqP993|W#pZNu~C2WIZv_yU1vn~p}@k&hauLX{ZHmw$^9PWVZX${"
        "wxDg)YSz~2c+}>|jzCJGBt!s<A}WwMAxKvKo8wBYQBSQ<xkgDHgkw4!+>CI_ti<`cqewt99HTcKs4wkn;|@?zsEOW2g$ID%R4(XB"
        "?{WtgS8(l<(~G>LmK~FC%3X~asGq0@4kp4fDUejFAc*oX2!`fVOvFG;B<BqL@tBG({gFo%oXAk<<irdDR_QeR66@lO4hiyG)FQyt"
        "=<qaLIqlVeTigP@gZtT+z&BdKbpTxzXxH32;gB;wmd^aC&B_t7I$GyZaH^`jKCy@)aN{x68^bz`G{nls@C^hLwh4`Uo)#<rR)26("
        ")LZ!BsiIm>7wrnVxga&`J#f4o786_ejvSR-5aWX0y^Yf_l!_jc1FrFvWL1l<tJ(GZGz{hJpMZf`2VL=3Qk91SMSl>O7_^FpGQ!X_"
        "A>^LC*Zc+E6%oL$fKyh1nF22IiUgea=qSpYGEi8KmXf^dD*2U*z6bMlOPy-vk36o?%f)3e!A%<3V=A7G=wqWvdl!w&X5Tc^vS#S{"
        "ihE|7L-O&e()2bj(<o{*2$llJY;(t8jJ!Cl#oiP)d<O067*%VKN?=TrWWyU#_>}<{IE+lQ%K{BA@T{Sg3`uL39(5Y6WLVSMXp^8j"
        "sh4o+Ug50>K)3a1JWQis7ZlUbW0CQR?1Hu@$5CV+(n5$E4;Bu-GKa2AZ>EPC;<7)QZtNx%jr-2;lP%TL1Czl21^6%(c`M+*dE{rR"
        "3N7w6->fk~z3;yN(B5+PKa)~&r)=--KYzOY_~D1!AAb%${dD{8)3I6d>k^4M>oYAB+{X0_8r*6YmZG7))1a53NfO<oydEck^q5X_"
        "tE_&vo_eSBk=lct__cQrek}SrVE<~E2G$Ai9q#j7pc2dYg9qvb0$?C}tHoZec+_T_dk%2Nk?J=ZQ(sa`uT@RIcVBXRn<)i!yFVCe"
        "v|KyZADQ2&3~i<!Jmb~Fj-rr$>Aa9sjC_3yV`g?-pAD6O8#~S8sDo{0cf((hTKqmQ#^8CaK4e30?AVuVT^HAvmyHsJzLjo3sme(b"
        "5*d#4=4W68NDJVsn5K=Nlbh)>deN(2=;S<A8q0#CY{v%|LlfEy-4V3!z3T%w{@UnAVEW4@gh>NCeAR<7VMW0+S0Uh+zpf&|<MG;0"
        "q8=JdIvx4cN=|P+q>3^7EZ>eod)?Pr__SYNHQem}{0g7rpJ3tBzb7$YT|%Jm2PNN+bVGAVNwt(W4Q{kxs6;?35<05tiis#V2KGIj"
        "-nn;Df;7P4+DeaTvNptyDl?kYmh8$p10GLgCM1b;v4`BO<C<1>tj#%_QF|1-153x;cI7y~jk<Q@D9Qfh-9qRD+8#f>#8c7TCfEb%"
        "ykgSkXg)?JN#Hkl`sz|nMk$yeo3yl0rC3y8K{X?tPiRseqw#Cc@9*+_m+~_*x8ue5cK$8in@czlhRY&AL$0Z2_B4pa7eFAF5fk~%"
        "d8Oz2;{B_8*JzB<zBvuW%k@300{jXu1>-sSoMfLFK)_F~BogSZu~W-r{KRl0dm11{)^{8F;0=$a>RQ?aKkc^5n!WJHk?;7qAtpEI"
        "j)f1fLz#lp_e4m0c{=4(3D`K;1Q?q>O?;=w8<{Qloi{#?dvAPNQ)+tOQdC?Vv3b?7abg>=o`2s95dIJEvn*hg>9gOLv-!Vq$#nz="
        "ixyQzP}}j{J9<%h8SZbBNs?{yyI9Sr8g%p~Sm&3ehqF)rxE;U8VdJ0437O5N-yQkHh3;SVMH&e4X&5dGiJKbJrpA>Xd8gczZk7G$"
        "bw@O6Azce*e4EsQcBn><|Lz>=IG=m2^ITu{;!Yf|t4N$~)E{?$oFACu^Lo$c=qKcQh5v7t$FL;r)v95UWZgiC--x+yVNGkM?~jjG"
        "4<9F2XQw@FN8`bmv@5UO#o@i$sqU;V6r=a%pVDS#7OA0(0aDiQ5jq^{Bzqk*B9yn;r}9g^J-D1zon9bE)p0*Tm#exqlkWU9(arYF"
        "i&1IgxDo)Z90Vv1K|tk6L4X$oLEztL+lto(Z{PecP)h>@6aWAK2mk;8Apkt?zz)_9002ue001Na0047xV=rQ9W^ZF*Y-eR{ZDB8D"
        "VRT_HZ)Rp}X>Mg-b7pidaCyBNU31&G@jWy74=5ixq$AOGdLONFx`#`AZDy|5OfKz<qTxU!Wbw?e5TtFLNB_OM3xEVDic#)5^@BwM"
        "i^c9@zd_IQ-nF%06;ErC%(rPKm`KtlEpAy=i;CC0Ny~!C9fx_g*|SaA?3iqLk?^c61pDR353DE~u_?<t8G4@g{MqwoX<n6e!;*Bn"
        "&C*TxzT<LdzyB)BqI;<UWtsP{;`y^}UFIw<vrNQDg=Bi}b=ek;s5L;v8^mfRk)L;^OdsFDI1=;xSt7RVHf^FtJT_657DD>?7%ZMY"
        "WAFz=|2=JBBfu)zPd~CmWNDtl4w98cwtvN5zrWm;b-ooMx%_vL#$_UyRMIH{2*j~xX#p63tfEm;K+IJtVj<a*;dY^{vb6ENkA)Yg"
        "=~^^xU9c6B#9lC*VXzOQcfhBXuNL#`^%~~=l^&q?bj$izCNe45Y8~_p>tl>+5tnu1*Zg4t2NE|kb}?hJZ0lGofbu$E^Y3XwO9Su%"
        "Y<MexRJAs0_LcAgwp=n8sX(^C^*#c*$GhnM+S3ayjx4BSGBooW0j!D{<696kh^X7N%#uh;QhLFxpD4tGHKD03fQ6RS!^U9^1L44u"
        "VwhM!iYgx8Z5vOCty?AA&99(p9quh^HcKU9<3%id3++tF4o-_;`iT&p(#j8PsqhrG`47)paZ}f2?R$ADB}&Di^VF_JQqG3`Gzrk1"
        "RV(mKKLOtaHc6Dot7adSb<`H|ju*EA&mj?pFtoGbIibTiO*pRT33JX-Eo6UM652_!+CsJ!nl{i=i-PAO;z<&Lk&33<Jmy6ShRHLa"
        "(D497L{FjL@6;^`QyKA%EVH%|e$WE|&0ZLKWZ4}o7=0sMn2#R`rXqi1Pb|jAdt_WjXef_<CkY3T7keM~1C_#EA5vh)3qfq?ByeJc"
        "wBQH>oMjCxO3-;=kJjRlAx$rs)>r!saFG!^Ia3gMmdeHwt#<rjTcq0*mf*WJjei4wtn@UPO#+b$4`DC{pf)YH9>07-;tt*bq$xGP"
        "<^%wu2h{lzU9dH0WYZZyj25u<kRLXm19{Bs$qa75pulZUPV`$6R;XSqHNcjBLs2C9kTFIm=hVlBk0063B?IH3A>J?mXXTb5Z{4OC"
        "$RmuiK&KA3Aq4N2Uw`9k856QA4}O$5VOO+rIOAqJ|Em`S;Z8gz>8*ehHO5uu5Doz&n6lrQu^RQrw!3YQ-SP=)Mm**jl%xqbf7Ge^"
        "7-KJ8Uwiu(hVmE6dA+d2^Wuocqy6>aA5)>D+*7}!XMA3iCf>p#(^4kywl3QWW))HPOh*>XcF(f`dYe|<V=(06YZR)+8CD7a6FQ<;"
        "Q=|(pqk_r!&hDn>)NEjIQ=l3uGf?}-4gu<1U`9RIQ(BwGq(j++w{W#j4U+PCo<l^I#-fm7xWEjnC2et6ln+I>^3W$vD(@A8%taz5"
        "^K9i%l%GVDFD|dI;4g6UC^f^rV=pKD>(7u`qDg0h%Z4=%C9Y+9x6r5}k7H4R3x>UgfD8mIpOVxiWPUgX3OX~%vC&oZLTQnRM<Nt`"
        "kto_+K;|lJj%@c2l+z0K{jz6a;d;zXE%;qO)YSU{0Hn`35O$pY8Nn^pj}@=PBeG}7g>lIQJ`8YczE^m@G#{+dmtMP2m@1}4D_q8a"
        "td(VNYm^8|^TZZj(n8iAgR~-LG?B8D^EO1pAtWpQ(Y6-b=2SyoRp8PI)^#-fxnrL0Yuc<_pDK%Eg|(Fx6l{6@EnpD$(YEGf!pp1h"
        "dIo35(?V<TOHg11)4f~zjOpDK=(3FiUa!%j^0e?ROcs&=K_<2TZE?@DG*N6^N}g9420mEZ>pnS8RCP&Ni2tOxTWPPlCPi0cc%+*5"
        "L!bF&qVN@x;CGl~0trr9#Epq4njGAevz^G2dD%9KfsSNV(T4Bo$Hx;0AXc@I8VN&*iLYR?z#7WXCUef<cyj*e!w(FBZrHOI&N&2u"
        "MuvmEu)rs%m(_xHF^We=g^r_Eg0o~so~xIuDsL>b>(#<K=ykX2S=fe{IxSf{TBgX;)eOK)p4JW4Eo#Xn(&3&0mr_8cjEF#K2Ks{Z"
        "(82@zGaH=+ga_9e37+)(xY*7)WW1Ou4BV>cc|Wvimaw`lSS}i#@P;$pzUW&ob@?EPYu=>y!YZS|H)8fJL);J%zVD~1vdlotR2PFH"
        "@GRRy<iku)7QB+Xvbj|0rpih{RN(w~kU=$2<iS7DXO4yr@MQO*NFg5*lw}DSLhf1TV5Pr7>Iw7JUbjXHaWByD9%}Gg$kA`2(2)l)"
        "f%A#W9kylOz9G)&Y(N6}20Yj<t-?+|>cI$*Vp<=;bm-DjwY8yqP&KLJ<$r$z!@_w+52K3~K;oUC9*}Q&`PoR~77RAi6zkQ<C%6mz"
        "UBS>A%R(TBo<|mr+k-{95hv{(<*Ce(=LKf}fYRc^0Gc4=?)7xM^~J@=ApQZ~MMv{Eh%K&h2>f#nZ#Ds^5(zk+`&43@Iti;yB2q=e"
        "BrNrsp!k$RHy}?z2jj!mv6c_VAcqlb{|BksmC6+N$`ejOSc?9c{TvIJM%hdQ=n7)`*r<}20#U`o6ojc3&Oh<taUPD^2s(ZkQ?t^y"
        "hHo+vZNbNgx~jpI`}b(i##iWtC+l78qlZbGS5mE;QIZsaGRWnJn%=u+_0sod=$IDRS_KwcOlf7Drpq6BCdGM0upxZfNA|JrbqO2g"
        "f<gvUMdEy03o+LwreZ2yI_Nm3LL<Z`+QO7%b>CTcoP23|i5}tcrn{UdBD&Skv_>&6AUlWzmQ9hN#OiAH{b8+qSaF6~`G8dxnw#AS"
        "_|(=lV7jb@oPV;_$1B>Z&Msjy2$YGdpx!3EMX6uzTF?%p5F6AE1Px}byhMR~;%4qJ$?FCVbC>h_F48`7npL~Itxsmwb-jFhqp>#o"
        "Yy?1K8J~2z1qQ-xCPOmgTMgw$-A%?2`Dhvt<oF)R?M-kHoY#=IG}?#<>=N>OIi?UJ5rRD(0yTMy4f0ls1aBZDu|p;{Nr$z+L0*;M"
        "jP?<po0b@wjv>bcG`ll@6y#809l6#b+_qUp*??brtE>6n_<Xy5`sS+xK%4bnEV6E6mFo2QTbCR;(WeVCz0>Y|GqbF17~luZiovae"
        "5d$VT?zM=?s)qUAIjmK62`MpdFc2kam(!uQy2IZ2ZFO7o9K-m0W4F_RuC=RF-uUv-PU;`&(2gNIZi-%9V7uqIEjgm(-e{ZBSnx=n"
        "((_!z|J(lQA8jGVE%w)v2urg_i;3=9v^J=rW07l7q8d^2HY_0-2dA)sEx82fBFSw3q3&{k5Hv2lu=X>aZxX)HpkXa|;$L6A{EA&L"
        "`~id5ct<seA}4IC1iONS&^=TQFq#8oC5I4dG3+N$NXB@&RtE5ovY=LpDi9eupE&}ng76iCA~M}VnG3&rsvK1rr20Af;=(xOJ_{JV"
        "|JT?4M1Dz*k?sL|s*Wm6%Ab%A>HD<gl6hT0gp&~NQId2Q<&zBn8O!dTqVoscB2pLpWPmq1GufouGUnN>$aC(9Yj}e`=_H(m2EQq{"
        "yX+BLNiHS*oaMbn($1&51-n-sdj=0_pRI=2mXtnSrr!}qHqrwzX^l*B<Z#iIsMx`{t<z@Te;(0$UsRQ)Rlcd9=s6U2_X2gJ;&8{Q"
        "ox8Y5tu`#^n?bcQXx4)`uL6<w1BM8z7(lrc0#a4kbWkA8NO0^lK#)=bhktPDI^j7tA0TNNN(^$|LMjQd<5P>>n@)1rM_PgibPFuY"
        "Al{mOpcR%?H--YU%A6tCCgR?45P0E^D+v?=3>ie`IT3iw-)Td_<)j?FY^gD%MN8A4jmNIqqxd>yp<V`I>5-8CbO=z6^vH5YohtE!"
        "%8orK%QfJ&YvA1Ks-@latqziFM>Le25H?u2s#i`1Wc3*dX6*_}*C%jv-g-p`@cGom5tX_ZFnvE;QEA`uq1$=L{}yMJ^7ojOD^r@F"
        ")_=YLJCjmGfuEk}ea5I6iVjHEiv8u=^|v}>!m(_IZf}k^A0_Y)v>b`#)5UZ-Y&e=tUGH$BLr9k^$m`H~oJ21&tNt3<*-ExG+joIg"
        "no*x>q)SC{Wuiaj)RQMVFD5%YHLViFQ?pKHeF|v0$8=7+aUjmIp6}<PqOL&P+LIQ7Z`aNM926O^ldIwUY^2H5Yk|Fwx@GeXINcSh"
        "B){%2@~l@YrF$>%V>wFF!Fn&dd@6xVt{LWR=r7QX;$sfzGWH<S+nsfz@dJk{nm#AWTZ!L_;sr<v`5kHvc7h}ZBGWKJHB&bhx}U^0"
        "Z2~{~KH)WfzlM9kLMaT5lIcJxArr+9Y4)bGl+5VL8(ZJt4qzOkMnYFB!y5{)4E(C9KM=HZ-LErsKIUM$=DcR#Eg8+?!}VXTufBch"
        "x?wemRxv5~bFu|<k&0}>QD_cWQ{IVUc%dpDD>yy?SvvhIO#_rZE;YkyOA`lh4w+6Jr%%*^b#Ox9cGklC*VixmY6^dN+_dyj-{~w*"
        "H<OS@X^$v=t$bJGf-q%xpAx2PvfjI)An7AFQ7;%Kd#LI-rl;`3OS<kOHM5n8c`n{9n2|ZFCqvfZm8p}S6wQawS~e`r;>K4{!I-h1"
        "cKKF&b|>OHyxFeslS*B60YPyI<ru#euNbeJbPK!a+d#GfYXRE=b?^HkR&!QWV(N!J_*UcZZr=a%?S~IHZ=%=#dUF%~^!D}5&%fOK"
        "f^N)!hUzH-AJJyQx3$SCphCPp4at$A2R1$G6Zp;cSSV6j`~cKrNT+^;%E8o-zN<N-xu@V!-&X5D>na3<gsx1NBS>`r6|}uO3j?o^"
        "m{ZlVhuqVAG|~}C-_+n2JbIyfXRg@8tX`P=XOVOtmeLG<KH$vKG>&#7U)r5C%ots<yM9A@N7{A(;q+GIQFG#psL)|^nCa?ilkKkh"
        "-G&LZTP&JC4L<Aw=}#k0s@)LH@weL!V%@FPnbgVU)4kg<jMV@AT{+2wsIec-6L5EpsyqJ+P)h>@6aWAK2mk;8AppTi@Igxl008zA"
        "001BW0047xV=rQ9W^ZF*Y-eR{ZDB8DVRT_HaA9;~Xk~LQaCyyGTW{Mq7Jd(q|G?qygVfk^+QAG4p3$~*X^R4F6ExWc7I9XkB`W4b"
        "A_bDl1(W&jbIze;%d(Sp-^MLskvuwge*9?8`I)yW_w3*2Z>B%6x5~3d_<Y4g(^M)KUK!2czjCG$UKY&C#yI7Ty=P8~#;uHZ3FrLq"
        ";P61zjj^5?w|=y8{pi|dV@)pIX4T8ODOCkBr8PAgUR7$zy46z{PS|N{<tbRa!yy_W(WNR)E~+b8*CNS{_EzNH^>2P!`}hX5PX^<&"
        "&Sg||cu>faIZ?_CgUZl+bR()(&YANzX48M+_x$jH!3S4ay6sMoj)Q<ISrne|qm5^@x|uSTlqAY!VhQPNy^P}BW#KaP_AZLqH|ZW7"
        "JV`iHItVwydN*3WQvOMS#e$aiXr4c^AoP8duY?8HT_bXE8)uCzXf|d)v2S0n5~gstR?)>4&o{cPP)@S{;Oc8@jg5G#UE5&MWRY!J"
        "i)z6;1y=gjuHEYEMqNsBE~4j6(R|xf@&ZN^mL$n#XCHd>ESQ46yYs9zg(|xacQa8GDm)dklJxW=SpHq+*u+}Y(#@7i3wuASl)mm-"
        "zN<SHTREkj7aFod?o5I^kQ7{{sRU+J6Fr>}*b<)N&({e63sxph3Y)J6EoS-uqPOqPS#T?uc=#m@a926@5IkoOxoPg%gHyGtgjN0?"
        "YE;O(hvXR~L9*!ii(~jaJpUfQ;+Ri1MnWtDqcZt0`F{%=YGoQ}DcgNqD=7+T9qcO5T7pZWsv?LA>Qb`l6#AHnS+qDiirF%J|C3KT"
        "kI0dDs{)<6C%vkngO$F@KzD{_vhY&x5w#Qeb8#8Q*wdyeCQQp)?CV_<U>gK*r-H~}{%g7&Ys2HMT(4XLz$<kTjc^H!#PP<d#>!IN"
        "fnQi&aBMq)(R4a}SRCQMm_^BM-nIiEC27hy^#09Ew^hYQks!h<Jwye0*TB|*^8j8GGmdJ4PoxO!b0#Z@o-b#7+Za>|4Z>zXXu4Gp"
        "T(69_%P4S^y?8OdjKeMqu;#v)>CkeaUZg04($}M1RY@H==_67Z6W-!f7@{vY`lCs5;baLF0NkVGuif}_^v;vLSG3_om6@>_s?2gz"
        "RjR;#+@KlXZtBg(hq}Ic#ionmD0(@cz*qc|L4}9ISFawg^W%G=OI53{M~F)}+C{bjUfeoT-zaNzEwzsbatojk0de6^kGLQO?eWL;"
        "oZV2~t|#ySsWUF=6eJ2LfCCj>wAaI~-~etJkN;dNOWLM@^dabFzn*-2^X_f-@#O5kXXmd!r2ILAK3I7B{^XaR-=|;-dKz0gte(IA"
        "^x@sdpWkPvpH4oUo~QhH6f>CFgo+%f!{W@FtmivX|1jUfEV=@0PNQ9s4WrmT!Nwpo+6iTXU!es{%wz9HLTG>aH*D+P!>!K7;;yM*"
        "axRgx!I|}&vH${*rYHbO$~%%brQ}R`%!DK87`kJiYf_?$gcNaNB~YuIEoMat16#W}8^K5+9l%`RyeW%F`HWr-so$r3uvZ7KBgwN8"
        "7_+*O&^n|l9$DL;&aISX(&eJrIwVGZBEf@QnAF#WvQZeg^xU>Ud*z*SJ~P)ed_u)7Py(4P@4a*p1&eTxN~O+$AHcEY%jpk19;nC-"
        "5O0m>f5DM&P5|?B{F@s99?^P}+p0`U0cQYC(@r9(>Fo-v5sgf>xfNFP%k5I-E16$ot)dXd8&on*9#o>Ou=Ja2MPgeW?WVYQ^@v3m"
        "V4k$|ORQNiM90(xoG(WZ%amVEc2y|dv_1_SZj_Tf()n$XV!ih)0lK6+__<GoXaJ31N3|X;QyzAuys9Ngq~9Ko7I)h-q$>#^hXH&|"
        "FYb!qr#o5$HkJbVrI<4g+3RdL4@a&A*d68xDi1(bLD6}9G5_xP_;OeIaK{pT-S9A)>n=#Ni-|(fu1f4xWtI_t2K_nTiMzjWx%(Pv"
        "d-QdCFhrPJi8tQ<-SH+IXh(PVs_uY+<kr~nwb<~4Z-G-k1`q2D_FQtm?(JLhBSQG;+#h*7xTsEg5m9t75|8S?$3y(=nI262gY6y;"
        "%-m|n{7N~1H$j&(N^8M@I9y5pBP1MFytG<W-BAjl2x7GXsbq_b62O16|M>VK!}Psx8LG+P>Z`A0F^sgM6uSgqO&}6H-6xwvX&H{s"
        "x>mp=0{x;w#smHZ!Qqb`S*zjt3XOo*m<VsQ`+l{6Tk46f?wQDQ*}!!Q;<#mWzrvf1mWYmE#Y>u;g`a*(z&OBN>Rf9eY^N18k{?`p"
        "=?0zBS9h!B3%k$*^4cwJ*xcvOpZPGWM8><jAMi680?7q2AhhVsLm)*j7w*|YN6E97@j|1L3F#(=dlqliGpK~t_z0nWv4Be72;eCl"
        "zoVds)$2keoynCoZ4*7;=H@#g4lb_zOSj|x&eKzWO!VLyN~s|i`dq?B?#y7G!?8sIM+|44=9FD4($wL>Ur<W}1QY-O00;m803iU2"
        "3ODE50{{R!3jhEo0001Ua$_%IX=ZO@VQgn*ZEayMWMOn+FK~HuXm4&`adlyAX>@rmaCwbZQE%He5PlDk|3L6bUX`qQP2q>2TZaHc"
        "TFk+=!Y~L%y4YMLQX{F@YLWlGBSlKIEVuKEMe^==_uY4Qq?F#R9<P+FI%2urf+a7!5+%1nNwODqM|fKb%bzO9i78YAC0TXS^21+T"
        "OCh(U6nra{v7#{7l+w$K%L`FAN?XE>9lrOyv~kC)&kv7xf3sii?|ysuxO%*Qzha+0{(k@dGg%UfO3e5RSkz$}3Z+XZnN$`R|3G1J"
        "@p9uYySymjnFwhb+-6N@cS^E8maI@Ey8Tm5o<#+c*RZ&~ydd~PM<3t?G=T#Ytp%bq(XiJ7KaFa20q;no!GM0j4et&Ct(7)}%aSO&"
        "L*3|KAGR~waBYBjL5$@=N$B@Ikb`)^ZLu4o%lhb+kGh2gVnzzy*j7UsT=n`<)gYV|U^EtVMtQa8ouT=0Mi!%<)m>u6{Se-3?`6Oy"
        "0<#<t8s*_oKx0YPTmRKXm6Jbt)k66G1F_Qvnc~;pc^~`a&SO;ke=m?w2M^Ype)m}o&|+HW;1HN^W7?;p;>Kh_*1-x}rMoE{6K`rT"
        "hHn8cG*%DFVRvyXn{t~s4ahP(7I43Br(2$4(I0wCrY>GH=CEcgKco9a$P)FakiHxl7=!_sfqz?&1<-jl#3gEwUV{u84$2VZyfRb<"
        "2=&aK6d*n4Ww3{ihOcvikz5HBaqFB)C1fh@mQu|KzaDR;^u>TCxgYAteK0sQl_-Rr$?1d^`AG)z(?BM^&iaKJN~=K_f-TKAio{dR"
        "1*T#wv}~Y3-_~N2>$5}`<XNjawcDH0fOR8N4v*~;>c)1HG&9rIHP>AhD<&97_)YyK(GSC1UmF@3X6tQ!DjTh1yOs8w&{M|(<-fhA"
        "$GmVY=>Oh0NPqv%L2TyM*Ab&awen;ZWdWo-#nq3PXm$C4Bk__!^T~%utg?83Wmr^g38X?I>8m)*u-9_e+08q`%UW1FDsb(p7;!>{"
        "anc}>9CH#UQUdwJg^{FjTpavl4GvPL92cSN`!odGA;fi1eM9IO_&4OulH8nCnq<LCW36O~)1fR}7TgG!F%ol>3<n(2wQFbd!FyY4"
        "C$sL`=w^eK{qD7U6Qn0OdAq<{ZpNhd9*D2NVjhMt`Z*ovwMjDkK}cR*6>8^RI-WL2iI}soI?>3_&Z==EGp6?mFeJd$7x1$#c})Z~"
        "+|2xLxsJ$awRgY-4--&+z-83)*wD~mCW83FSZ@{y#r<35Y?Fvw#=3Ipj(FuWB0Zs|*WC_8_)!Dcq()Dw{?y8plBxRG4Nmn27LIM="
        "KTt~p1QY-O00;m803iU+fx-2f9smHsbN~P$0001Ua$_%IX=ZO@VQgn*ZEayMWMOn+FLGsUWnpt=b1ras?LBLc8#j{Q1LQw2ZQv|9"
        "njYJ`x0g(y1Dx1Sj7=QF_U-{Y88mt%O6p9jyVH*u%ai%<R}Yd+ve}X*-aXs_r_Cl(H(4wetBO^{DozwdFYEH7EW|QxQ&G!Yrj2aG"
        "s;osSsye$)TPc2cNv|8(LW`=LiAGjwjc-<UxfR=TDRXfnv-PHJ;wXwf|LpV6vTaq?tw`%NG;O4Lut}Ruo?V*XzcyuIe$;)tX4AD<"
        "Zhp3MTdlIZfB!L4zX^p!ndfrRW@XU`)!@6bD_U8r236YPu=?d2_=#B3yUT1<F4BB0x7#$viN@t>m1l)anpLZN<Yl@{79GxdnN($&"
        "0~OlzGMC9JE7r0GS`{sVeEIX+pPs!Ja}n8a6Ggy8kvO@20%Usq?57vdvX=3p+*UB1NgbW1Cm%2HUyo0|PEIavAJ6{d>$_<B`Dbrm"
        "{QcFtS3kcVbbyBG$?D?v&yT4c{%<K)VwtUF(@t>a)3eV%6Y#mo+6~Q}pT=b+i%C>pMpKbCFrQkcTh$alwQRe(&`TC0y%PnM+s;JP"
        "Y|^Ly{+DPPZ{!E{&cflQ&RUt^8q<g~ah>L!)aY24aKl!9XeZb@UUu86nNaJQ$cm*b+WFI&kVVtgGD(|7md#(Lc_XLdOA-B2MEGt|"
        "E@3U^QP-|czOqSCmp9E}LJR-|iL%RNzGTK{RFN~93G&O7<OaV1%BrWxR`eu>pIJ4TT3Ds^Vv}7X1Ld-r)MeSuM3HV~GZSUkR$Z&H"
        "1(A9#ftR;gku`0$5dV1fMw9@c5b{HnXNwF3EzirF9A1ATic&1n#RlOhp%j2t-4(#ZYbBrvHY=`ytYy6ej=H_G2%}^x{zq25L>^;o"
        "V6kt2%;x2*H_7uCFMoXY{>Af|Kz6KU)8z76=JUT0>D2rz+97#xhD08aD!H7{gr`;;{4`}(FC;QL5Jj9)AGZZiwSZULFou!K{8~<?"
        "an_)Or7hC6Es3c8TYnhh@ic3s_&YM%i@Gjr5C<ip3_C#vL0%dk9mYh8Or}WY{#moz!opuo?l<Z-YZ~Ceq^uL{4ax>#pGCmtNCh7e"
        "y@coZ@&Xr9OXQhW0F2QTmhW=c$_8F296^79u)=?mCi^Jwe8{U3z*(&D3ZBq%&2TpW@@88(pFH{ctH(3(g#LNV|4au+uF6<5NYd^q"
        "X#=*{$n8d~s8DqkRR)3zvap_M7UTCg4lyY;0;XM;&?rEC-dg~rYKJ(Km%mPWkh<SY<g!?|oB5N+PyZ|)3H*<mn9_^4J>-bE<V?wJ"
        "OjNSo`b(n3+is%QMnQ1b#%{Qk+e=wvQ?vN$^_{PGuQPe0^lmK|WxWKA@MtDl;J$)pSk+})!rnQ5T^6vc&8BP{`fh2SA4`<#rV&+l"
        "395#4si^X<>2(i|BWtlj&DeY+B&cn%E<sDeEOw2t7A(!fdup{yf%$K6wW>7qCWel`bzpX2i{Ef^*ai^pwy@F9#kob7tbox95ZG)P"
        "+4QQ*TsjY!rJP6Dx9-S(aGo#90=FdR8PE*KP9vfEcisW6%>W2d)dxXj9|8ay;VNl>F<tyXa3r#NGMJ29W^Hl>8>Bk`a-+jt5>&6-"
        "GhM-s;*Ekl;vjXQR%K<F#u6vB4slw6^ercVe7yz>2L@jljoH9Hv`t~Dsx;>BlXIm+@CodR#Tpc|vok2T7Yf8B!o$VBu(qh`xA+b7"
        "#(yEJ*NV+~h<-bQwU|dAn|8@MfyAQrHYDUg7*$D#r!JuV7Pab7i8u^*hEUF(M;Eg|A6h@vn7#*FDGohD6Jmm50A!sQz7^)lC0Hz*"
        "ZCYQ!!zh|gry*cAZ=dyj9hS%5p}vNnt&Tm9kM&}PXYA?w6d8!ViL9ZW6LpM`V7>v25C~U7&V*$X>Q}hg-rmi~j!3TLu9?5D!L$rP"
        "v!tq@$grK?7ACj_u1d^|aT<~Jt03OtY2c=vvn`iswJ{tbh@LCm2QeH~22Lwf!d+_6MxtvGRv4$<hb+(1dY9O^qe6k7A-F;0dj(|i"
        "h^3Xrm_%)urVFdAvDQj|o!Pw48V!5c`e`3mvC6Z6C>?ko7`Bph_!5Bn%oy}nLO?acSg4t;XOw{6KI9-1W{Q9o(;vW@)e?NU7;lF9"
        "5?UbZkBnjPQ|5Cmud~M65`*ovgA67GP;Hd4O?&Xw<R5CBT+2FJWsB6`se?T&GNa{su>Bf1A9kN@)*fuyNDL*^<+@I{2|C$FOalQA"
        "_WNMlqO7-Ro_(ZgCRGWu-yNPU?zIWUK920sRqW@2Ho7^Ce_E6UP(DvEj?^4E-@{m!uv0W(j2>?KTL(00cgY&;(W4u$@^pO+vLvy`"
        "60TeU4`x6Ly`k;vF&%#edP_+ptU$Qg-3a8Nb{c8uskOsfgYQfR?MLe^Zck`xog2M(5okz_-Bw_m?NLOl=)r>r;#F_z{8HSal6z=b"
        "KMybNeu3wK&Dw>6k&rXN-Ep&`e*Ba7B6u(YB(Qos4aZ`>pm@iP|J3bE#;~@y5y&ty0dDJrnoY%gF2L^$P?8N{9d*MnCK%B1d;Ah("
        "Xe+697)xR}+N_@Qf2L%CChNNFD)LU}Fh*dCjOro>V`XB!1`O$qht<CVBkc`ec3HmEk=IF6$wk8M!XV7{bCK`JdB|leDO#CftYaxq"
        "#WZ4p5jTt~3ll5<2K<=11v#eRnn-I&%uHcq<L}~-u)0>bkYbMplW7bpbjjh*K@?Kaj&TJj)Ky__+%pw6kRO0XjT3U>P!n?R3wo*I"
        "P2X+G5}v1`kl_Aow;j=3&q|NB%)s8k5JRjTx(DK2o35pC>VaNO31C(!jIhL*0Q&DhD5OGOmRBOru5cU*A2b_iqsEBsd7-Bdl#RDn"
        "Fyw@P7$XYtPRc7b@GU1$9}2|pC58>alzp!8odB4*1DYJbRyVW=)X;nsT}=avW9`F*K2pkWidD95EawzIXe7aaH7R9WW7Jm$#6Yu="
        "+dJnT=Mnz$g<dI0JS~V<U7l}I2FQ!1kfY@{>p_kRIc5d%H-;LQ9foo+s@DVYgPs(NRb*1g8uo}TyvF7Y^k4#?3Wo~8sBX`0Lq6~Z"
        "z?c&fz8?q%z=u(wodP%1m1-VjP+g$9jO-(xC=@$HGXssCW+t|2k*&ZR1T+T6Llk-rXvMTc--w|sv*y<_E80Rf&7=n$>cw%+CLD^u"
        "PsHIla0A?9h4F2T4jjf`!#;#tc8p3ZM>k|CulpZ*>h#3tSNrqjAxGgHT&gWNjBKZZ4q3xCz~k%ziyv6Oi8>${hcPueaX&C}`Of!@"
        "7;?h&$6-QT{rF>6joEVnMDB)<7I=I~>rJj1Zt8q?rhPDVmc}|pvrddA0n9Z35TgZnDoLvvFjVpkOl=%cFD)(%D)Z<9SfRX<sJhN0"
        "ZWdY^cKBE(7h};2bt*d{Lq|olE@(=$dwa8?8NzhzI32YM&@k{wIebT=v`<ouQ*Qg5{qz1&)D>4nc~eCGA&C7sG~N9@Z8{vyVSz^o"
        "iv~BR17ThoF~qGS2G+%=f^`@k$m+m-G_{T%v<nbY=lD^(skUzK9m9wiKtfySneYx_2CdOVV<icV&-ZJG+lbxEGc7+e6Tw>9+uG?Z"
        "j(T^B6gg%6sVOn$gaw9BW~U0W9vlZ1^D(1@V$JvoD2u@A$m?tgwyE5=1)(2qr9%MLy)v<%k72@kCQ&-(OzEVt#wx<Zn$jF=jyAW2"
        "mx1%wvG%L3AYy}yt^)xaIg%Lb+~+P2BDzDBx+lOJfG|50tsT(=<M^4~avVFSRS1;FH!>d>RG?38qjj_WicTo18hr+^k7y+@KpowM"
        "!vRCHjDd`(qG1+KAN`K0r<27)PxN7;|6wy~rgNW$MjrK6*&11~$UB$|9(ImtwH_!bCUO?Q^ajVIxu-dWqgXID*%Abj#@=r)9hO}D"
        "7#u!Hf(ep|XxY8Z!@XVg*-_t*pWH)tp(QuMDC!6sRuV%Alog~6NaUErS(dRKTxZ3NLQxG)(e*n7%;GL9IxtQkGZG6E8@w#G%sYES"
        "2}hd?J+Ul?8<J=?j>iZuIzYdi>nx!03E#4%Sfmv)0%<*EFe*DpttlSOxvf-4G{7NyChf`C=z4yds11o`vHU*q7zT~^??lwekcFfB"
        "i;Y}dp^WM1R<Dl}-E<!d;)m6rrz`L8I|bH8F-W$g1sI;H50^f$6zCa-N5nV^cMHGA{5^2B8rc7OOr}%u#hjk=vk~5bz83ah97dnX"
        "7QlbZ<{rjySSB$&677?1)@<$U3y&sgu^S<a@np!{py?9m5yJwrgGXWE=g^4+e(rIWNK(o1@I%KcPVoFt<FsgQ+$5M_vs$iXjbeRx"
        "blg8toc_RJ&dB2dzkf2j5H+(R(aee(9fN>3by=+0;QXhCploH4s03FxVnVMdW-@BBBGMy}d!6mUCDUjOYC!i`E^QbK?dZ}v8=q?g"
        "Wu+a%FO4s}PkWp6JiV<P@*0jgO1#~(iGV~b)2h{uK~!)e?J`@f5)P&()Bt2=%+VZMK+Y|6CZbHcP>*1A0NuQqj$_@@8boKdTE3f&"
        "YOVhxu?3vcQK1E!|3)kv{QPm3f5eI&s+6v1x~d{G&J;J0<K_XRk`k6))MZ+on4ni+FxBRq>4E2?L5g5*wAa*yW%cRvq=&=x*?D<r"
        "aU+%G<T$6{Y#tSmP%Im0vXsrD&MGBm`ZUFQ7zC_o*6YC_a4lpAS$!Qpo>yl}<7Y>KO=Mevt)bT1Ugihlb=hVZ1IA-e`5|4jLg$AF"
        "N^qWuLSD;SbPfD#QA;vhK=q(SakD|m=E{Y*2TGZt1XdrXvGU+&4T#J6h0i?dNR6A^`%LX#hi5zn>jnP|3=;(lQl|DPm_{4}ri3WQ"
        "OWP0ap6>4HXy`mtf4tQ8i6#O2O*0#))sqjU@Em&uYv!MF4$d=G@*H;BOaD(l()19!*UGqkgDYg}OzcBVCY9IHa?VZ7$)OYb1AL(9"
        "=dc00@c#lnV5G;^KfnWf6{XsrY|QMc8=B%YPMf4Eo9x46Bum(>iLfyXrhuvf3474dDhd2p9IUe%2D6D)m?ejMFPH6kAR?}8NI1;|"
        "!@mt{?5BQ1wU}o<8U1#!R{{{wdrhAid1eV!=)F&lNEsoVTENofH?fb}&us1PH#9lq$pUPayj(lz4BwKt4@Sd-@3J;H>E8piFx+5T"
        "(T_nBiutD+^MLIXJ(u7(%cY=EP^iik?Ijr|>X`jT<&)x;1B#}svo&x{E-=3v7OCBob+gH;IGE)74iQ)gs$4xKS5k1)L@Y}JC#m)`"
        "Yc`@kCBWL)G^>R=EC@q>U5bT7=1}>kh<jB_lX;A1a=>Z@4v$H0=~iowT5d~FY+<l(R7{epZnD1QW(nIa;Cxe{>Jprp$OK6jum?3K"
        "I0o1|(5K8<rhNGJ0u4pViwd)T^QvMes;#APth!L64|Q4)*Pn>7PK&f&awpSfrp<dh&CD)tbZ1oUS5{|%8oJ6ZpR4v)&f#`hSrv{n"
        "KwPi$@^TVA8te|R6IOi*VUX4#YMJ?IdYK0n3^lYw9pR>osS-9tz&=%OZ%i;4^|P|)z6Fz*S+FfdyOE;7eN6>uIZ0AzXd;ESBvgV0"
        "3_P=32}PecH&UOi(m(a7PM=n6LY<gSy%z)xDowRGj=oa?(zY~k#z0W|a)#7Snzmc+G?+dDjVX&<+s-g(svwVZ@mK)`HbDgs!ZK5G"
        "TVPe>WN-q7RZ@$VcX`VK3PijqjkeHWJ!=VxmKB{9rd4K|YCedv_YHBs$4pzR9CzB6)ef8qY<|B=+>uC?9l3Ubnud$!0g`h~w}X}E"
        "@Q*OeME{}y=#VT*D|n_XCUan*iT7`>?LPq$Ee64H6XxzqLmefKT+-OVRmv-YpC059;V|T4{yn!L+4i$${V(U@iKBhg3#%gjh~&6l"
        "-Mp2oKY+0dux?-kHN_0%TB#e(N98L7F!BTMam&SkD$<qiHjq|LJSMqY?#~~{kpZg&ZFJ;J+1kQ!u+~uZjJpao@Ptm|sqmW<y+~o`"
        "l&?M|$uFru`)s-&Fz3E^wLA!&XNp7#0=7f=rdUH)Yr)>Q2HU0MDlJ`(g6WZ9C2_q=$konPW_SQS7TVTRN|#G)hRzdBMwc*Ih$8mq"
        "G!ZmUR7ZZzwG)j>bp-R}-_-A=yms<*$LsZ}X9Df1Y6iC3qy#r1Pb>TW3CnWfJ#=8WsIk;GvP-wQ-Z-|hP3dwL$Bv2ev{-jA9NNf{"
        "q(mV)cd*Shlb3cTq7wFFx7=5kLn&{<RPvmF2Xew3dScRLLLKpb0*X>@+O}%WPER-88jS2UI6ink$*H!$Pj9uc_)s|p4=?Uc?fvuw"
        "o1R{Rg><?Fa-RP9>bn=O-@SMjZ{50PD$c|-<u%yz99xah5I9-OLb5a95<rocTp;@9mSM}b-I@7+%Lsjs2nm(SOJ0OVLFoisXTSLT"
        "<mALg6xT0q(v?iG9B-nR(5?^UVnBT>$ZQ-|^gtZMjM_Or-e}`t#n<~72LsKnK08JjM>Q7T0sEZtT|T6FgcU|NOT5n7O?Qd6Il%JP"
        "U3mr8zO$6$DE$s2Xg5^+F5Y6a?|06=Y{c(gQy_wH{_Rzy>w57SS7>KN?~pjT_5lS=>!%L_RxlQakzt@)s>*qY@n7Ijd|;JS@;eUc"
        "KD?O1YrT<vGA_Wuh|4+93#94MBSz@aqce?>6&c1VBdy(>-aSyyXCBnCMt`Hi4-C7BAG%Ayid($<;TdrUCAsbN98RrmQg5p0Oad<r"
        "T*c_U<}&SaHY(oRNF)Q_3}Mm4zAK`Ta%tTu;ho_`gH)?nJj?Um;cYYc#ZE2STIhiw7t?6zWl3tM8Dji*zPdtPQ*iy13<gXY)@9!6"
        "R*e3x45Cr4FU7FRf#U(rugUtw+j<&w@*7#huLe5E`V?~|)vy5k)$@knl8YIO@8A9O)nCns%>Xc%kFX3+W`BC}wZNkwgDsjTE!sdB"
        "kjQu6pcl3?MM+O*fBDl_qAN0JPJxgBLCg{YQBBUF;GDw~;sup+sKN-McZGBS1SVK@O@JK0si<U-9WZ<YUsV#OR!k;JV*yp$Dovs*"
        "ln1dKBL1xoRhg<B?9rcR0<-CZ&weed6n5qvjm?@*1#bqpYZ98k51=64%5??`e0Ne}*_ery^;KXPBi?Cs8|1%t($rj$BhMBmVB9f7"
        "a0cu*I_`uX)g_>ril;cw4Y{t3`08p1dGO%D1M#C3Tie&(@=Lz#m6;uxQ!SJtGFfSFUp#yM(+k|Toh!ZYg02{?dAcU6*?VV~1J5=D"
        "9Ei^T_%|K%GYS7ly0nG8^|Ir#Pp7~$dcOhg?)4C=H@r3L5{#-V+1i{5YUa7R`py_)J<l#n$@f{<z=3yjuz-MBfz|UJw&Upgi47Oz"
        "<F8UyqBpigUuq88CIg=aOXWBQ?5KbUXJ_J^>NrW<Y$R?xOXNtJHYjNQ^3<rGMyw<TK9>6I1C3Iqmpy_)QXs8?`|uc{#+$_SN!$XB"
        "T$D;=SMaT}>ud><BfigC+)?P{Ti^Ae;+-fk_7FN`i7_grZ~Ek{6CJUq@=+uStV7v1Xf~kPvW8VWcgY>0jj%wl>`+0O%L*-qAi~O<"
        "*k_CvpxfT0i>sIKXJDD&wy1v@<)zsHN%ljxO^X*G-q=bk<swJP|LN6xb6?_+sI)Gu;MEeqJfO?WqVmsRn>>3Ji=WYG?-h$Fv8g4R"
        "$sX6K93_e)Vem~w?12wyr-DbIo2zt3p<vc6yvT^N<sSnf(<RItjLi5K=Yj-ZT4zQHnD8Ol4yM%BWaDU?$F3plU+SQMKBTb@+pGdF"
        "D>{5ky)62Xst>uc>ZFGPDe8_!R<(Y#x}PtXUiJR5TXm(OE2~Zy;$!NZ9TQhx1C8q<WHlNV>;~_K(;3>h$**fGmSab|L)7!4#mCUE"
        "w`$GVZx3Z?$*K@x@WMVjEO_iyeUwe9;)r86>5Ws$U^<8#Vz?c_x*>dA9CiH5QT_`Y-@ZZ9YN3O7M`J^!8fMM>yZpvN@m*1YR!LN*"
        "$3$noapk)92+G<$(>*iHuaLL9(Ionw!I_NDLPHubpCi;3E`EeKc(<gBU+~sQosO+9`@`Rqdhaf)&fDc1B~8jM|Kx);{Sx2dFPj$n"
        "UcEsw!(oc(u3oT@u2Ae#onrjWczYwbvP0J;B8>vKUh-{ZLA1}Ee#3N&K+fB7LK(=#Q<KBifx2qD1+l)CFuwFc3D_}x{!=_n9zTAp"
        "|L=CW-@TePpJB2q!uK;Rh!*#}h(Ldcxd+!#@)fcw_kzIlO_n$6CQHs%KJJQ24Tk&dxt`KmbYFw<3XCYQ{RCu2bERLaDzYi9s{<xl"
        "vsGo1Eoa7zomob=zI}tPqy(><BKVYjJU2C#j?=GFRH;OeE~)(dGy8&EeaMYB%b^4;ty?YFBNXVS#o~2oqG2a^hiPw0>x2UD{Lu<)"
        "CY#ROA-!4QuSADBbi7t=cKWtvE4xWkVgF|T+}yOyt;Mzj2mjJUZ2DRK`SHnO1E%c)FC^)MIOeJjETpsPs*eY6`x;g*!b8cMxJ~D#"
        "Qk+ahdm;9tud}breb)dD<&E;U^`($H?aCverJ@Y$4ATPO1>VFp*EfbmJ%7!6Mq@Uie><*f(8}c7SoogY|2Hi?({g2d%8G@;0Z!-Y"
        "aEyG+{a7TOI0r1t{Q*;}B$O}BMD}#}H5j@W%+@=+GNNY9@3#N`MSbTurF3^q+G@v>84ZI#IMZ7PojWI=R@@Rb?AxI<ZFxTr&;Gps"
        ")nCKP3S3d)3y%tWS6T~X50z!puNnHI_Y4iRO|L?{eI_|@bBZ&j1%9LwwcpoF;`7D%uonHe-Hs-P_it1GP1+=BK}Tl}EdOR4w%|2>"
        "{wnOhd&xSCF^O8kYnT4_-;(5<1*=M+nX20%3AXnTzHX&&=i73<g)8*pn&kJ3z#WwonBKRa^h>4o72Ag9nt@N)TlB}{3(tc?Bwf#q"
        "bI<-2GUP-49~2=?eT7Oe?<7!NVom4PQiM+S>de$=u;nsy&xPnt^#13a$G!U!;d<HCHm%b6!&ug=XQD^+9izR8$}7-Ts#!)BEB{jU"
        "?_fu0<!M(blNDobrm$Sc4uOuGz=_d3Oj$)bF6i(#UC=?~afa#-t7lcF&C}lg8I9Eqtw)}Ob9pmbg(G&Z*KNN6;x$gt+D$IOYyxoV"
        "B2~<m(#(@q8K6!x(rzb`3DpF=if=Y$x`j7x(E-_Ov}7^mflQ*)XnOvb*q(c<vfp7VvaOyTX|PwNHrDg9SnDK6cjYj>o3e$?687}Q"
        "84wC;d$s(hfnxHuUOJ0|?s}I;=TrBe#)`8J1~%TxRYwOKfV$#M&6JvNKo6AqHopZSW@g|CHKMa1KBv^Ay|Xddj{auGX%Y;y&#W4q"
        "spEp_9HHUt#}|>>E_TAcBUs&S81$<~Bts7koa<O=-25mzZG&zI_;;Xwmx;=#QaVW^qI^E42d`JM0>$TCW~E3o*W~x!c<W&A=Lm@6"
        "CG%eQ_3xEzZe`PeZk<PO)Va)}tZUB3TlIfgisgd~#w-|1iT7;c7~sMd&D^noPai$v#4$d?f?pR(l5K^{mIVF%3Og80x`a`jd(^ku"
        "(I3s}ho|72D<gI=-8;4TgK>5ISYN-S+h80h7yY>ku^ay-mNA~UEEJw^T>64QQMLO>fp&Y8?9(;rN+B322>cNgg%Ore4wx;n%3XRq"
        "-E=G(jW+QfUl)Xqr`s3QgewvHoGrbf<B^&1Sg5pvb`A9h%Hn7By4%7syrDM}E9GXMfJM1T5=-hp#vA&h7T6Y(5R<06<X;WNz~_-_"
        "Mi%669dM0vLl9j_i|L387{OowlZakMc1wDlDhwyl$q81_0BDllH|j6EjEpkq%l6OV-uk3<1WQjDsh@+<`a4389dG~wxoL!sd%B>O"
        "I8j;xdZ(mS=26q4CT#)4A<VVjIAOhE(!C`AJ(3_%=%v&0KC>C?ps&!sM<>gIzE8Jfd1C}qzN~DqDN$w5&%Hn+)t|1?E^l>+cP}($"
        "%b$>Gzi$DdlO>Jx9|UA>Qbf}Ve#UCW$g5%*b0Ss^AO>LE+oGu>O52QK41yDWWY6pxV}FU^HuCis4(zKhzl4wPno>=jIFWzZ=Ww5)"
        "o7n~w(DHls33KtCq+tuaK5@u~7&M|kFByLCJ6gKE{c!A_d(Ro8uH#lgfRNtLu)%9MXyiEL&`i~H6=ev#vmD{TTXme`zHtv$c!c5g"
        "2F^JPHL{<~1{22*^NPhCVZIIoq=C1Un&H0@WXsi^{sW1?u?eMSCkb(DlHmTGB${1$H~;*z{{~P?0|XQR000O8001EX(G?yp2^9bU"
        "&`SUSBme*ab8=%ZVrgb?V_|G(Wo>O?FJxhKVJ~uJZDnL>VRUJ4ZZ2?n)ja)=+%|T<8}R?YQ_xo9o1Lr;=TJ|&hhB5ep=d8KE<vuq"
        "bRlphYDTt3QbcJ#;`)E@`yeHf;>@no<cE#{??`%leDCpnlbR$+etac*byP(qhpKHvTM8-4zWrG>LXP)UUCI7Iz~iB6_9AclZq#3k"
        "rWMDwQ1zEN{4dnK%10c%Ns{FGv**vMW7qbBkb0QD=WX{eT^yu7)YZ*&{a&}tba9l!VY;}JeFGEg=g)wqBVeg(mE%-e7>IA$u^Cjq"
        "7KJM1SPw;&4+gp%n|x^7dV;O{G0SCsPzIobselgq_#Mq`p50Vso6CByjz_sE<RCZs*t3i#mI{^WYTw9V?A02?fVZ;_W?T#1)zy&Y"
        "ZPNg~s>pf;vMcmS<ZW5jRRj3S!QwYUH$&H=Ze@`X5rEIkGRT`+WhH>AzU!-II5|Q0FdW(@`)QOw*n^!zj{wkDKdVd@$7(>#M&vWa"
        "_%E-e*5dCK;24KLx49mAT$U7;c=H{|;fG!}x@`NS>Q~$6&jfs6J;Yu$s;~0ww$jy*YE_qO(49hA82sS1xB^N#)u*eCKZJm1*s2t8"
        "y8*VFT+zd|AR7MK7Gn(;`2Aa4c37uBtCDa0Nk26C;_f%d5?rlZV$39nd2`bi55Xh+lvS?>Ab5h=U0yGtd`f8yPy6=1>sLZH#p3zq"
        "R@S58;hUBo4Iyo3(DQJM1QzJjSSxfQ-ectD*5RaApw<mNWgYm}$$ZkIyw{*KSFbPM`~j8+ml}xaK-Dvtc+Alt(%f0-p?5`j2YNJ_"
        "_9mA@ez0n%Q`WnyANo;&oV3X0xIYX5FVkQRid?I*s=)`OiLCGBgHG1s`-f(b_kU-ol|R-f9E!P<y=Fq!VtD8j80_(=nnCmQ{y2En"
        "RbXwZ@7o>_eAhP0hw!-!#u;@@{sqlZtKk9`Z>aJMIf9-45ok`(kHX)gZHBCB4qygDj1G05t8Ne}^Y|92S&JWW+0FOAfjjOs=sI1K"
        "5Q*l~v`sw0>ZSDEnX?@0njVF|hAgibkDXzyYVvvnpC||L4mabV^wqbr)@tP-L&wo|ssMybZ?f7%t={PA4Y<72jwgk6U_fWoBShMv"
        "Pb1W*|3_bD2?`_Gl1$dYBP0PH;h*pk1iTChhR674Inr4LLg8G8z*hAE4!n0-X+7SMdrt4rl+?cMA3_5V;F`r9@NKMV<skjUUq4r?"
        "whK|QJ>Wm|2kI8=DEpuu(!Ra3(BQV<YM1DC>~n=rlWTExC6Y$n4`_zK{Q_^Afd3v23XBD;!x^7kn@hCXYM3rDtWOu{B+(-Q;mNgG"
        "-wE*^MxDd4n)&H^RjOX0yLA})0`d<W_fF;?5Uqsi1kftR4yIM|XtIKT(wHzohyv@*M|3fiN_;IO*w$SISwgQZMo7m1jbQg+#%qY1"
        "AkkiwsHgP<WJ&!8up%&RxLcJKeE&-YImtogAHb;e#^=-#>O%g%yAp=`4!9^GgP~L*adhOOR!vGHZ&ZnDe!UVGV!A0y@V;<w9ZX{b"
        "R8|!rSbpe;6q1F#$}sJbdzG>j)?(kcV`t?H3FUt%)rq3gx$FzzlI0<)0%QpJk<vom%Bn`Q7emW=7v-uLV#&91R*Kjl^#RYT!B6Z="
        "phLR=QqeMKUPt>3@CGzfH3c#Xxyg_w#RyZNp=CyjCQ$-K+Fh^wWQ9;`h&Y2;<iG{nIVAWcm}BXmKQ3?K2NrHCEJbj!ii06{@D#a="
        ";I<!cnbv3EjMbe3v1*?Iac2RIxt#@(P1}H^Ymji)>Qk^0-aq-b3<R&h?(_m)V*ml=QblSOW{u*zs^#&fkm7zT?gO0;FozU72xp2u"
        ";<JU~@eO<ecP3q1hyYdW(_~0-u2{5a+$Sk;9o~WzM|%GwK30P|2Ecd=p{+S#G77g0$qeaGY0cf1nUlaxss~bju?GLz?t7*66^0RO"
        "{<SI~jyv|6W3`6{Y|6gWNMXu4@!3Zk(l7+!4W~3r>q-wq6hbgr?;&qR`v%vqB$T3S<H=Rw5i`%fW4LSa`z!Hse#q3ujX7e-o+@om"
        "drWVPvB7N+2P2S7BSJD$Q4nU+5MZtXNJkwexset$g}SF)sy(8=1%*%zgf^^C?aI2MtIBDOx%|QL;vdAT%gam50WJgIXo1s@3CLuR"
        "JSR@4f)i$WM2g{&p(YvHL1KbNvB?ne$xZ7((XXf*u_++1;Fr|0u<+fi(S2L1WJO_e=s`kBKq%bP%1M>dL}3v|R2o%>vZ;X}kSWyJ"
        "jIh9biJRk6T!rEfLn%<WET{bNyFLYBW3k5zV}(eB`W%w)=s%B!;sY3{3Y!Ql>L7@`&8U_$?1D`8`thm!0iQ9`lm<Ux5=}X+Eis=o"
        "3k#YQmA~#dzo&F{Vis?lo8C+IvbN8#MIwi+ZR&@|$(WtmZs{D*v!|ZcIQZD}?fe}4+Il9uu$F9c&mbk@kc1XZh`6Z$^_jzKCo38j"
        "IJQWh;j$4jvAevsa*Xy+9vQnFqaZyFI&*37+RbOu_i~~y7#XY{1<@&lE~+cT4tgcdZ^a+QOVTd9MY1Fh^Ku(O8_!Se$Jh7cO|>7}"
        "QD;!vWW4^jvackp;^PX($}p!4G$rRibNRFK<bE!PcUlPN*?2;xyx?@scaT#sEv+Ds5yuc3Y*rM@)@Dwp#G5ZCqxeF+#XhOPZUoj9"
        "RntL9#civiDoGIB{}qD-{^6MfX8~h+i86&U1)|_V^%mW5kWkQppKAuF-SR`*YAR8%>j`<CD)4UGL$DYis{&gF&8=<~%?Q<;7?hq$"
        "Izvn}s|jRnxO9Us4c<lCB<ZqRj+N7sA%*UQHR`~kIpIxs_Lw&TCJk2~pJ<MO#Vzg3pd?HC)vU^BL`IxEZctVrwurQLVe-?T#tLL{"
        "I!O&^wb&eT!?fXi$e71rRXD|uNu+b!i@MPu0r8@s8+>NAb}&ywPqeCOSL9^WnK8ta<^QLfvP%QS9PaEq!AV&$o$yi)wHESM&<tml"
        "=e4Ig@~k=p2pY_D>gs3Qse<5^;f@~{qf5^dBoHSESV!l#*{(BxNJ=30&mrZ!d7LEllwm9=qdmf@PT%>nCsbaWYJQuD-wQGYT6X^C"
        "b_(L<KlKY{wAa3>*YZZy>#3YiQ_GFisGfWQhMt-<ns0Ni8QC-=JXG0H%V6qxhZ%B|f2-pOtuo9+9NXqYOvRu+gkVo9j9WJ(!e9aC"
        "#xXBFZjEMR12)jsx1gGsG9)n@Kii@92aqbveJ@J8s70BaUidw24V#V%ZzMfsVn^-l=uo3D2cp6mNSPGfZ}&WTB7c9&L6UudYWop+"
        "6D3GNR;#_tA2M>NG~2Hta7Tov<xIK+W}zCbGVYV-+&qX8KQ^|N4(VjIuNv?ezk!Cp;H*%24H;vRVN<~N>@z!koTJa%JBiAG*akj7"
        "H0@oJM1UWGBN(%NnQ5T=D3NGwCHg7=1w&J^&kq^3PzdRCrX<4*nA|T`|9sRToC+rT0Q0oQx)XN@CR@OGIE3u{tbWS-Oct^;HF*vW"
        "s{)qPa+Qs*H(e6}4s(tNUY8@b+<LsB4SdMr+4saI3iZ_-&dBx%#^Tb$)FQD5Gs<3Z(bLh!IqwsZCY9!l11K&va9G!aNgN<M-=ugr"
        "C+&ou2<I%bt_3}LU;@ptk}10Y?*oh8-y)&6ugu(L7s_TSYtt!R3rcpmr|L9{Y1anrM5i1g@T(0t7%r$xz$aTTYX+_4olbg2dC@f7"
        ")CaPzA>ly;0Gv)`3-yvCZRH6WRX^HCxxwh<XQnkCx(RZ5q$-+XB91VM(ykUPMQ`1<m!>U4?hH@;cw#OVZWk@eIbU#R2nrg5a3hN%"
        "ou>+be#6eC;nM3~q8%<;fX>bH9mu#aY`?SZCnMv!J`~9Rr7;+vx<TU!d__1bmq5#28-5b*cx+?Ytl$2NwZ6jSc-mKp=FBk!enA{1"
        "V>t>xpWAO(Vu><DblBDV?Z{c%0EP3gBYqN%!OkRlgGL_J+HTYGQb#<C{eT7<Q1Uf9939g%{4BK%`$-ij9Qm-x1yHJuh$0L)vydBz"
        "Z#JV--VxU8v#{~nCA!NQ;mMsPLC}QfA&kLH9;``QO_R`!^0MM{qn9P#Jbw8s!m(MfMa}aVi;fjy7MGlos?7{dmNjow2@W=(!);sh"
        "s`GOWo}c3yc@r$`cX8a@E%H)#XS5<1dukV)7#&`~4jnOTNWG?3k@!cJKvJk%b8F>_&>LZF2?Iofn|RJNQ`g8lrnF!#?^~^nH9#@h"
        "*Rms9z*BG@KHFbd9tw-$pzTO!wBrM;lsM-Bz&Qf~c-LD%`d$4rVpP^<JJUpC4=8TX+y=*Smrno=oc&_hiSDNh_sQMmhx`=EK2A<e"
        "dpz6^PLkvu=ZQ+NM8wU*p!C-;oDKj3UDRpUVx&&_Un)F8^`GBf;Le%AJ|1fyJe!aA<Od0EKI|&mY>O%I_VTIWP6Pb2){pPqzj%S|"
        "y-56N>w-%YmcJh~BJhp6Tfmz{=oES^e=Mj8PA3pQ3i&Cx=C!z@DvA$-xu<ScVZif>(^;IqyNX4}z}=X|`s|}mzvR@!BszVNufF;N"
        "3^Q*aZ1_eRA$Ox1bh_X|HV1WIRC}eNU^s;wrF{zdB$y|ojGte!p9rzkoi($BZ8EY$dbhx>+pvDD%dAj!1pyFz&!Sp03*=h%(0))2"
        "zAUgfz+<2Ed_Wv3JVbluWDgkak63j+&<V&A^iD_(rINTAo1%tn*W}qS7_=B&0G_JEaNXig+WiH+e}Y>gZSS1$TfY1N##L5(`@X~5"
        "|9bcKZ~y++x8MDcegE$5H{YKc6@B4@n9P_Wb#|l?hc}vD-<Uc~lfY~@pSTBs9c5FMxZOtwnby-`CN4!)j=n79!Z9sAMpjZk7r~9r"
        "^=lfuDHL5ry0<t8iyHX5A=!d4(LM(jmPA`5U*HEjcg(fJg8u**Iw6A-IeCw-YT!#bX`o)QMB9<lSeDg2Ng%;fLKj$3xkb7vaffoI"
        "GTFE7zE)q-0$#knc>C(#C<`DwkcZ)NYp>T@?GKNjc>M?p%#!u8LzN~Is~dbPDp@acJzNw<9BDi)5s1W*9vQ>*08AWeC3Wm-h_{>f"
        "Rr6jZ@%;*uWTEId!6zq=VlbDsl0mT(Z=s4UvoEk>Ly)nvSs}7%2N0ZkDM5HsZ^C#d&lcU|&Bi0mcZ`n47Ww)~U>9^O+<`1GS)c}*"
        "F|MmqgJ1h|81Rgo;}yHeU1IKfkj72}NZLv>k38@jsTH|+AAY(W8#B3O`(=4R`(~_ae09y7+vK;-Qce=qX3u`<;uq1tbNtx#g0XdB"
        "-GTc?o53Ay6VLrZ-dd%MEpnO^Fi*t+7$09#C9z&|Yv$na$gxfn+F1H(DY0ILL0I{`H-Yewql?i${}{;S)PcNZonm7<lchB93<lDx"
        "h%aT-t{Kl_4C#(U^dKLchAsj;tRNTRJy^$%eT>F$tb9?KM_r41qSzx4(erXIl1Klf?Z+1{5W(6x^BRhb<wFHpS+{qIcO=xzvU_Y0"
        "MKCDDPExs=mB4VkUD}&_o|Q{62)h}+P?S;QJ6X>Mb4}$psh=gAlpJ4qy)|!|imOv^oQAqfbKvW5bCdm}J=bdSlF?vq{r1!3bt#7l"
        "PtJNuU_ieRKfV<BHmSyap)akrF%%z^vh3mOD~C!CZI9g_p>DCsjIISjoJD3|c<I$m4}#)Bv;y-xHq&gROq?j%V#H*KE*t1b@f;1g"
        "9Je<}D@;6jH~q1{a8qj@J|AJe_ItXCNzk{Zt4oS8@2of@F<Ey@bnbfE_Zaxau%qlCUN^jRF*iiK_Q7-jd>-oQn-6Q$d^=f;*CiOa"
        "e@uVf2k}iQ;1PGMIxsDKYuJ17#5PIn&h0D~vn81Y`^@Sddl|SR)<f0+A%V`<&b<j&{<aHd3?}+;m$;=8KD4JQXgYUUOmB`h>A;mk"
        "74H^OjMbj^&LGcPE;<0^2bBCvR|Ge_w+9|Kd?NFFJdU*$<7=;H)Zq@JW^Y1>0RmsB-{h*QQ5{kuW+lE9uf5f!*RO9O@DzMJJu?oS"
        "b*VQFJ_8K<;Xr=g^w^z)c#Qf4R&XZ21Sbqz`3q@mmvK=LGg3AYh@pNOaYLo{-UmUX5G?D$Ju*cnkOM*-uRZO(n6qYFd<S$B((BlP"
        "EGNuN6j|`aeh!_-*_tVLbSM*7jIZ&9yO%5R0(_q&ds+qCfzF8@3t~?7bYERUP`TWEMQ3@sH($}!8+-YNFW=DRQQpJlOZPxkU_e&R"
        "ff}2zzxCJ0Ne5g%$1!0M-U2mO-EbbpaY;LGf=SjHpImaC?si^@QwH$)Y8oYvM`TAxN=BrB_URJyX159pVLL(AmKiofGw_?-f-(J8"
        "n~i#|(nd0tXlC{!e15$ZiF-u@G6m8YqU*`nHeK7L$pj1vLm7S)Wl85~I0^?ZV(10Rv%t|=9!2lys5vt5_-|gMwEq_5sOB61Ju-&H"
        "Y17&|J}@6loG34|1{6jgrVvEBZ(0ojOnY6q+gEt>jRH*c)Q8moAL+z~X$nTgkjx?ea_zlEemQTX_yvMF=o*wLLH2vj)No=CoL6}s"
        "Vzi8Wgj=!%b3_b@#KmCtr&AymtTQUT=V}P3s_QAcadZBKGxO#QRe9>6J}rHVc2|ARWw$xbIymD1$cCM&(zjvTg~Dx&`6p;N&_fwa"
        "MS`u>nIj`<YOUEPZep65`U?9*b+X#rp})%TxRp~MCduY~TQw;H(GM)J*{op7<iDH5tLqpaSP1nxK<}=SaVRhTIhk4xEN*J;Sbm%r"
        "#etJ)H{AZXP`{^N@38$-y1?@%_QB>){hnUFdPh&vLg~D(I^^#v%ZfJ7vXwg&H&-)h#-z!`g?-(bei`Hng@y8?y1Q_Hm*d267VIw`"
        "KB@2p7xWGuTgml%Oiavp`hysBI*C({HEKn?jpH=5di)_3?&<`0jdn!3M&_kxN=`sM>1c(c3cM!|sYpHQ1GDR32^&N-SJ-0kbR#^w"
        ";Y{VHL9sJ3ZZ)?xatUg@`0KH56t%<Vo`>fPqNi)9oQMrIg~r)eZOtG0L9J3`8MalkjE;O}8Ja|v*%O~^AkUxuFHlPZ1QY-O00;m8"
        "03iT=t?Q7%4FCXpEdT%`0001Ua$_%IX=ZO@VQgn*ZEayMWMOn+FLGsYZ*ysMZ*qAqaCzlg>yz8I5&!NP{|_wtg-Rrz?(I`XnPd_>"
        "<7p>xe2)9!dgVYQcw&ztX#kW?cJF`h?gAje$2~G>zqFjOPXrc=#Xf%61?c<!yR0a9;>ofWG52J=<{KtGRwUj}net@8D!DF|w_%El"
        "jOE#9rn0i|(o%S3n&w%-J;AHeyL|H@^nL&3i<d95O;rlzt(jcs*{XT}r!0%+MeydQtXGvNV=m>(7pW*WUR>rmk8v6qvQ_LEy0;(!"
        "Q>8Fe2~#Z2nUq{wil)hsR7{}+w)-8tqI}f1%F;4s`3>J}SQwXu5-e5?r}w;KnRtE2HdW52_*%*g=H9=_y6=}pL1SD|9AeGylw};V"
        "qAW5XiL7cP)YH_H9M>F)hI3>_KFAqX2i9st4qm?S;9*sj?<4AqRN2fj)x9^Fig>ZjL|JTjVOupdYD=0H$x=?TP%AV}P;r;P{PzBb"
        "*U|geAKzZS{pIINZ|Qx}^nRM%sakM<;RU`c>Ve!BYRy#^&)TKUEcLV*INKiL8(1ML{K>S-HR4;A|54V&qv0%fJc52MkgaZ`_6&3`"
        "ihOf$`M1}zH%Ku1fHKVv59fdJrw-)>KTZ#i&(2o-D868a?C5x&9Gx7mPUiDh$EPPJ(Dmx{bbi<wq}OL!I>HFP;ssb?d(@Mt;H#{b"
        "B#)t)eLU(gJ)EDsI!^dJJ)NigG-0PFrxzzj=krr`eiok{f(++}EIm)3A&KDf6Dj)FS_p8EgZHm5XTPHQ;yHhHI$s^e7xUy0URLMH"
        "@#-i&hPAUZ9?vi0qj-LjvW}|n(Q975_^t`e0DSo~FP0xf&B3Lp&55QDOx`ZEC|MyDD`FmH$-<LL7{4T#8Z?0RwwVOLaC$_UyM*W>"
        "&g+Uv4iUE5WXfsHq^#XRiisPpbTm25c^30Rb_lstk&H!FX?EVfyq`;e#wGnRQMOm>vT(!iK~lj#)tTT(BOfHnS#eXd8!ZO(_<90g"
        ">pbBptr_$OTr#x}8VyW6z;ReVuuigAT~QPKcio+3;?2IrW@8YrX^maTCIs2aEw2QJ7z+HgQk7gB900nk>lHxW=D?pA8d;}A6GA66"
        "0>7*ZeC*H59T&mGXg>9d61Rmgt@C_?3J-*T#b%%9vx_L4U4MPxzJQmJJQEN0AwLX#@E6^(Xnh<!TypOtw!Rjk6oFqCvaS#<c@k-1"
        "3FLeugi#AD(*6SOU+#^kab7@=<jTstTf=5x^a9gcrM3dVqMBYD?sSdIM+PH4QM`|HZUwXRhb&0~z3{ff8ph0jMeo;L_5KkRKPh!q"
        "0AZFyt@KuI!2y7cVY+T<_BaIrAL0HMiE<Y>(LIHEcG^G5563YgwRsCJ43ym8pq_As+oIeR?v|^$S?LkAz}yf)AdLZ-Dd7reyFb;H"
        "9(!$pBtj@ff6)Rimj+y0G{RV7yAS}z7`_&{k884nc@7f$u7eUjx#JKeMNPTV$WTbDRyp59fTW7R#PT6PbNR?JuaSv{x-nMNZiZ{z"
        "bjLez+Q>?{lV)6W1WrbnJ6P{`@qp!d0D3q5!Ip4pLh0$&V9A7X(Jh*vAa1I)G}Vio8eyl7Jd??zezAe^S#c8$+%)5P(zrGzo{)7p"
        "!r6&X!bd?3B8<(_?`LnWzno7`=J&o8@^QIrn2qRqD2&!bqb#S#XlJa;Xslhp<BJ?{ZZz!jaFR<sK^Ggkf$bzd4!5?^2}ws)Ha49m"
        "2+|Hke<PeYy?RX+Fr5xkm31?xUF*q#W+j5g!%cGF)7I2)jS99gT^4_Qmvck!Bosgnj-+8{X_6})IJiqcG<XGDoRez&ekN`X5~;iA"
        "bLeWrBq?UiCPkU_ls$1b2bM%Ot1EZpGi#dzVFgFqfvp9vaD%vBG@gc_hD0=~jY512btXc>hgge3MbQeqgKR*Pl@%`nNVCfXKw#<D"
        "DxJLo>7giIgTLpzw;dw*P!$aLGfH4YKpN{xBNX`T5n!lH2KBH+1~|qng69t{!(9c22WY@yK|@HVSW6DqHOvp3fJSyxkX)*#lK^8f"
        "oH{bDBO@jR+q+4OUSd<fKeo1tiBC0G?($5fTkU~wxw8~#n(7n!k86_^ZE;|f>Qnem6JqVI9V5)Zf1;Hxpac3F$?P*<9v;2wFVzv5"
        "2t%l<t34_uXlx^&pLQBOr65ZLa>AzI8}HY}&`+7NX9}0#;FhuP`@evbZ8GHORj{d+BD={7mV3Xw|10cnn@z14?DpO&FIVy#Fr|mP"
        "mae1noKjiD1!+%d$X_7fq9`je6-9yMd5Y;xsoZdzdS79^0Pz|XT<uD6yYN<JnS-0XVX$ylUC1Fs1~v-uHx>erUTk?V=}~sLQUdF|"
        "TM@s*<RyzY9DpNnq=xj}K!cGIN;O?85kRc=s*{R?40VojwWm#?sEeR$>nk6t6sTZlGx#w}GU3x!roF~Yay{)10+e0$&Q&_Q1wJvW"
        "=pdu4RaGl0e|r=v+i>Uw51cza-SE_l2=&ehf!v3hf#(fz#&3?DP~?<>8$x~UdtY1TQYBo7FyY8QFfcKtG&E<J8(%Gs=kx0>=ZULm"
        "+SXfcuqfBv?EFF38K&f|i(WcF&bMKA6fR)htUwc<&gXO8tgYNMN6>-)%%;+<8*Eg$cw#aFQ`NQ^0?HCnn$QxP&|5lQe&7Y>kTaBP"
        "c653cSPmfpEx7z7T=TmmyWt2Ipk-4B4^UTgZ$l-d3^@na-||S6K{M&oNUT{=#uaf1p2r`akKX}~$-$0fh=x%*vU6zx5L<sP)tj;|"
        "5+lcDb9SsXXYy)*CKK@~2RMZm*m9uIlZakP=*nuHgBBicZXq5a67gJ0>Jais{;0gQ2TC2mo)z-=G+`g-kjm)z*FwYZbnJNJi#7Cp"
        "sD~m*=pmCnKH&2ljKJq<oWRFo1q9;s!9X;2d<rvK6F!O?__WwTdisx{M^A7i_B|P!dUHfD@(<b#wEAhX?F`&R8cS}u$kM&95hir1"
        "Ck)4be9^z}`+kRyUB_6LN#!0?KfjB4r5vUH(GVL6i|(>1oN&oHGZ<O$<bOb|1rX2?-H3#_oGHPzd?Nqc*43e`av)s!f1kSo(XIll"
        "t{jI{C8SHxHt(jBIPV_QNh83xHHm7(fOwUY6?M;skkc@oVq;YZbJFluR>YOVo+>t=xT$xW70J{q@n8zmRv2n|nbAWWz#-nBmfp(8"
        "1kuNI5P8{?vo9<Ea6UgfIe|Ip=lSWWfBzh;Xi)*mm0KCX)&qbGylYTy1ni?G`RzLc>fbEAd*KvdYY#;xy}{a0UoUa@4^RukHEWV@"
        "+f)Hp4O0eW_=)%J(n&6U*HUqxraXRCCwN1E(aw|M3h54yvXc>ajhUL^+~}`l6f$WO9sTu=dKTR+F6}&hmGu%;4N1Qh84~?Sq+`&8"
        "&!4>?^!4Mx<(m&)jc4IB6_m|;*ug0;0fs!LV?)@|c_KZA@mA$r;hivO9;&PO7B;Hzxe)<2!w^M>#Mz!5No2v0RZzMoLHDvYuue85"
        "mXl$sI!qDU{I5MeRCQ&T$%%7ZQ2`ZXB<^9M0L6+V!V`t4m+g+`xVp<yE=J@rB&xjS1vt7r5c|sf1>cEpxjui#5{T24j!i?Jb=PBP"
        "F&fOZb40T>$@R(n;ta2WF{dU~(8<e`m>O1N*W%Z0`T7wLFB!N-S;U-<4vqfRntON^o3vjWAmRduNDgH?Os`8okJU+L>sR9~+E92V"
        "WpT`2Y?>3Q`)3`g_K$jzqjM77Ee@=dx<EioeW-YNbZ!SrU^FDh8~2FarU!N|XMeS4XRuc?p(OfQFxcszH?<}nOx}E?@Ni+&+#yan"
        "K8Rlg3=dmdVzMk<et@kyetSP+Q#=RK<|Kb<#cBsXx=y++-!O6O8u(Mq1Va#MEWF|%6{qho5K70t&}huw+<{P-*bg@l1eXtg#Nk%E"
        "HmLPN&2fAi+Z2R4_?E+XY|cwHNdHvkiO15E<F_kY9bqU6s!Ly96JkS#!69F>Lboy-WY*!geCXqmpeM6?E#5ihH!AEE#e*@nI&OA$"
        "w@B9fM7e9fE+GUlu&VYIer4i2I@WV<yI)$kcS;MR>bA+HG?#vcNWTsHn@<e7gwe?5gHQ0WXdT(@YYxOszkTn^Zg2G#uA4YC#+1-+"
        "J<%1}dakwr5IH{X65fciu8@ay`tB5T$Fvwlq0V~I77uZ;_aSz@`gLR}QR9_3rTy&^whoRSpmzmheeJoJZKicW10jIvywiHc;{fC0"
        "NV&W6U3c}b!4(Hi+#5h46!jH<{Rp7Yh{+@Kx=h;m8XmQkyQ$fLiKQl3XGuaIEPVN|1O?qnTNg)&clz85%BP*@`MwHRXAH6!nxpe="
        "lXH1yN!uoua`;V-9Wd<aF2B7TYt_)Ub$!k?l(tJ_sweZv|A)-c$r9f|WyJ;57MK-Q^$Os^Q@e7eKXT!BbOH4JySLxCyB#PUVpy8<"
        "T!^$bN9^c4!a(|63A<Ijd%f9J?U;H8s$F*894e#YrTOdh3>7+(4Yb#%<8Ida7uTP%ns05iX91zM?euXU!b5mER%owRI^V~(U*kNP"
        "J5jcbI5z;w=OwJjZIG_EbzCcxV?UbDMLzd{RtuxS&TXglK5q4fuidJ9e9c|FyOB8e-TwwqO9KQH000080000X07LUtx@{Z)0K{kj"
        "04e|g0CRF<FJftCZ)0I>XJu_|VJ~E1bYU-YWpHnEX>@OLd0%61a&UEXE^v9pJ^gdrHnP9pO#cTy-ur<{B-&2XH#xV;y<Fn-JeMXJ"
        "+v&8U_&5{^NvtVS<%ez6m;e3kE<Ol=pd`2V=5*IYA{UFrVzFOXAinQ^%!-1gUc>Gi@B6Ir;wo8Xx2*Q!BK7htVMWc-v6t0#%O+)!"
        "Z@r3b$~tSxYU^FTz4GF84KK5@2z}rG>Q7(&DO+#Ks_|BFy~?wB_q%B}VUlO8Xw<V`>atKjDyF_Raka?u?sr?|fGT8FRaW-Hiv8NM"
        "8gMSEa_uE$p0lI@a%xY&y)N6LVU++#<0ek>xUQK3fC1%k!UVu3Zjf&I@+15ldmr11eS}fp<4+tUfzGo<nZ)^$t=Dmwltoj;Nu$tS"
        "C_cS~v5d9fmqom(S7qZugY<<-Tg5mAi7NQ&Pab?eo?ZNKesy*BCVKth%~|yB{Po%U%d^WdfBU*j*@s(J<?-gcNZGx3oWSF#&Xz^o"
        "z#zm!c8_?YO<CsR3GzEivn8-nJgGN%)<iI@0vO8DsA5T3r8T@A4H8~17I{`ce~U)ao|kbN(SQI4b?DViJcr&ES+Qi*rpk(8PQYv="
        "t8S|3XM|H!1DbfjB9umYFpOf`tjZ$#wT)psTZ^TaKfk|v^;7ib{MGmGKU`j&zrKt<egMY)thfil6V=<IiSMJkxGI3=S~79HX|~q8"
        "sBWR-YAZmpqTT@aB0&Sl^^!F^IjF5JlhO^mi7P%}4j^VEy33kXR129B9ajvvUYBi^K#TIOR*!*OSqf_qSfM==2%EURiI6aC!^y<|"
        "(R%S-am^rw_YzvLQ;q^w{snZ13gqwYoJ0<%-HSn=z_0Z>Yrgdg2EqxFN2|mu@heN(hUp3gTS`5Gh*P%kq9`js5F^0{5>1vb#@)}V"
        "RO80a6YYDP=jB~gu;#9;Zl-VJyk^?lHB6jPc+q^@uzGrOd>kF~|13DkU;+5e5F9Qfe63!LPEEo?B-fjgW(n~72qwKQ^IH~-EJ%ZT"
        "Xkz@fV0&W$_}()|8*^Nuk@?Wkibg4b>AOB-vUNc7<e^mQOpx*i%?)`B5Revt#s{vF094qMuwKQ_zWJXZP*|RN{{Q;$C$*l(HH)~!"
        "jY3wSJO!gsxMKGlk<Mb&8i2T)lOrz>-2u)J7Hhct&DR=a3$G|i39UkQpVdtr=;P8r`m<-jk=Vvr%{(*?@5|;bD4O&P4O}2?vSg#Q"
        "$zX&%vbDuB)X6dgD_q~CSrwqFD4KeDRkdJxh}fvSp`TXEhM}4Pl?AMzk#Tj>E*9B+;DZ9Lnk>%k{uZ!);|AM;qAy@5pvr@<ZEg_$"
        "YU_B4AMD7o;vxJjH>?QUEvRQ7J&F4C=t!!Wa0!UoIauKJ(SsC=FGqa-jvj2e_~MU;u{YV8m2ES9{>=c;$ct+b+j;{Y1{)X{Au5|V"
        "2bLZWgV6XD`$HPHe>eB3ixv1^`K}HK!wQ7QOIB@h<DE_wSs`lw;N<w(bMJ|Vf4SzxP4LsHmjfAua<sn@h&ylhazF_k&@q1aQur~>"
        "TPD_k#FCbG1-TIOHU--mvHOIvl%=*nx;~tT;a!zAZ0C^Xz_6SCE9`5d3~&#7=g@|Kzq<N(M$q834M>je#D<?P-qAn@V6V6s?gDwu"
        "9W&@}2#Vy0x*M1Um_50Mr&I6Qj=Ad3kFpSIg21kWw}WowBF=K~H9?Tc$q{b5j<1t?;d}?EdQ4SI$_QLV(An@8xUj#mL&fZmiIZ}r"
        "z#RdXC0;VJR5(QNUS!9XR3Nq199Vvq8oQ;Btk*a;?qUp>Xa_v^IZWfMfoWrtBMRFMxSK4HeBoAgszm{OM#i4$A2}vP!(K6nF$xC*"
        "xgUwXL7$JrIr0SXPcj1#DG1(xGpBJxbGj^FOKLdjm&~sDZ%Yl=j1^?^l_OB!WSb2b+o_Bb<qP4Zy)O$ULjk17c*He04_3MU=7aI8"
        "f1&4q>raeQ6HqjrYqp+)GYpoq|DH=z)!;CUxHI6b!XjR?5gF#%5;*CfU}Mz_!Q-64QzSU>kAO6I?T4X{BoO?df8tt2ILe6CNxWgU"
        "9tIkSbKt8WpIm4;W2x8K@pT_SF`E3DisI0~pZH_%WHg(cT!-LRb@Sw4K*hpEo9Al`76X5FJo!sJSzJFnKL*n;uoD9imjBe*<JN+<"
        "WDXNCWNe4o$uTNF*uhfCEbqTKR~|K6>o#w)O^$(J+0+i~LQb+8Jx3S$nA^yTmN~^uwZ>K8Y%bQ=%+IsprjE)gD$1f_%QlZI{~CRc"
        "lSlNK(=r2ojJ)rhMr#Krxc3Z!a~?NRkZp!2M@S3|bzF}K>Gdexgz|)xkpdcdFSXF<AQF|1$5@cU_B{@d90RNqSkU1wWhU4%J2}0!"
        "hvf9Z&HPCq5n6496V%bVYl7RNZa3%yV6F#w`{>z~RD=mw$VHi_!3cE}D(=VeG2s>#e6AFyoN`9}sEHL}oTh=JR~xhe%eBTBoXb)u"
        "5*Z})TvDtSm}9IO*%0(0E|$)A=$wFck5-8wx$~~c9t5K&{WVKoI&V5ct)Mk{+>)SnWXN%Okk0I_2&ec^%6Ka24^u|Lw%<03$qO5U"
        "5TZ{-hCpu!bgwXhl86A?i^mI*|M{3K8<M!Ihj~_j5Bf;_w5G!tFcfKECV7<{jj3xXLqbssyCw{ets44M#Pf^fD@m!Ad=-vXKcyUS"
        "KT3K}O$&No=p^vv9=h)QO_^uO)<5-pl;ay%u%E8pzKDcc=G|hxKSFOWOQOrSS9;*=698cb9G&JfKm=k^kN`zWQwI<e0k6_5X#(-n"
        "z_Dlx3UcB+N>(hnfv5=WB@C-!&`!_jebK$lq+Av#(_#xFp!pE&t~cpDx%6Spa2*7dfBEsvPm^~Xc5+UJc#*N{^mF|rnEmr}as71U"
        "i%C^;ifFV6yv3=*WektQWmUGD-~^rID%*gw+;bPi6!KPpO?)Yaviu%{k$RwEZwiQEe2$2)LZ3*O*#%$>^x0;W*rq{Ev|<&TZ@nc%"
        "Yz^-2EGovla^WQ_Oi<I(A!J$}wMbz;SIn!+MRONdl#JR`<!y$h$!ehc0AiRvh@-D4^W#knvnltNy7s&uh#82|XW?=8%7K$(nIM>R"
        "&=pHF<o~>-%@oAl*2H6OV(AIj0qB*l>m4J9;r`B{TMnf^uhYrblO&IU7Qd&u0>5D3g#s&(+vvJ$+(D-uv`$&UDBwE~u`U}8z}C7X"
        "G1%X;et&!Aq$!bGs7p3m2)D{KTfpDIjw9$=F_w7-Zw5J-)yW}Zf?%DP5Nh~8$tO);!HT3si-q}7IUk^P8QAR<@B`7vw@o_-+49nI"
        "9cRT(8EOkDgob9-)>)@hC`q3xm4cHqmiBMV6}lXg@B42{%>M$}xZ4J8GR%}tw9Sr-%PYA8;7yXZDI0ryPX{~}FT<@w5fbn{5n_wj"
        "2*aRoBQ_)Pc|WqnI5X;!voDMk{DMSlz_aR`B0**($c&P0k{h9)@}wv@)!8f*aY_o;*2>a@F(}j8fMcU(!X}bW6rY5y$=Kc~kp9V@"
        "%>c}W6<cequH)*4hv^cFGwP#=UV%CSNFP?}rpj1@-i9Mn=P!%aTe5;x5L5`+m0~#r4#Ech<$<j<g6#m!%6*C%uDN|&WJMb1IT*-)"
        "ehxo}Uhvmb0q`#f%CTedb94%(@@eq-^63an=70Ozx&Smkn2bhmLH0w=>&b4K5?J~frmHY*AO$}xKx0$j0-7i=6D2A`f{yi(8WRN2"
        "x?69>KdKj#*sgQ*+F*WNF`Y%Qp&8IemN<l#n5hTXhSuTjl*B()mZhem^o(4x4DD2ru%OGzQ7oiss@LdPDOEJbov0Y3wI;z9rc&&="
        "gFG;Fe1VF5i!Q?wf+bJv(sx=)rr{naGlp$X=~*awf|&6EX&^yZNUgqb43caumxpcm4m!_T0YTa{h&NMvTKRKrPJy!k#l3iVBLINy"
        ")>hx3Pe}Fzuvjvn=9Kb}3M~MRv?LeW<ed$zgqRRwI~6EKC1l)e!gqymtIrYpU~o$=qwkXs+sM7b5$*x_q0hYr{rOW&F!V9y@s<$H"
        "-03dUJPvHU;FOw+MJ})q)XTbd<=4dY*jY_JumN~-L5_U6gt6_6Viw-WJ9VTXWjhD(aPp6cEA&9i{`fl-yA2aI_sv0Y>+~S7sy+za"
        "1fBIm;J5>RD9S@eBhc;zN5>Q#r9)tL;rh~l?2raDTPW}Y$z5X&*OUMY2>x+9j?NW1(B%LMr7-xHo`#Ug2u1K-=TBx$MP8B$%A^nv"
        "xq7q1yk)ln=xDn(=tc)kS}C!Thk|8te#g?LjfEquD72j*M;@_1*a-TJu-vg}%yg`@RWd1L;&$i)pK_z8o~j4y0f;U!Ycil;A0!gt"
        "G{_k_?FQ|0w}H|t=p#5J9Y5V1sFtD(#^9M_nOKx-_{}_(WRT&aB>Q}nWF@CbpwT$-;B($Zf(d9XW}G9}V}pJCje2mc5s3rG*V3KX"
        "OApT?2b?xB#OrQQB*OR2E;Kb0<*Azc^&_hwwDzz(Zp)pbIB8mNySE-Rx}!)OgIezORY$mYj8&H3!Xog>Duz&udqY~3xafK!C&x!W"
        "ErNI+1Xt#wDR;Ir18ENkH^DMk*`km!!nizA9myp@v~+6l;SQX}@=neUah*Tn;io|+e^G<$JG&lz5eVbkILonQ)bJ?x`;c0_RxQ)6"
        "7$9^1*ZXeTVSW@FltLuhnJl!llCy_-;d+H-m(Kl2#hz3H{EnqdWp-eK-?SC6V#BIQ*)~ZDVSG|no3{2SLAoikqN%_2(vtCMuCY)P"
        "^EWKV%G+kRz>=5i@hxoNQ-igxM%IAm9_n8)_WN3*kW(&88m&qWiq(WIdQ5E^RWqb>lq23>C`ys-4g_E*%1HzN^~Kq%t26KF)!*Ko"
        "dFOAv_aCmjv!Bi{uP)W70#y+0U7h`OHS%}hv(k4vu#4%agL1Kep1kw-S7+a!U3eca&VP7y@w4}jv!BPCcuS?RyxX3_Bmxv|Q%9+E"
        "pk_vEfm58yT1K50muK(JUSDY_-(Gz9L9xLfg$o9{tSp$73lSL6|K(rB=CeYy*MN4yEJ~MoMZ4@8bImZOG;|bL@*b@odP6%0Lw|9F"
        "O#aYe{qfbiPhkFozhb(+o8|GyF~NM3ga^)YEnuxQtpi=|QPJ=z#nr**kE^a-f+AQi8p?LWe<>*5hl@8LAAkE<K<V;e4S$<42(%>?"
        "cyJ;_d}xdau`!A|zGY%HTXgq|k%u^{NK_mhbDkO`r0sgsk#H!Jlf0~1kISkBz17kg^ES&<z3!c>JHc<Rdu1&<OdoAZ$x{uK%Rs@4"
        "K=X))O?ku6qp6|<;|zJ+p=nb@T`aPtcmzDyw9-A(VP%E6FoSeR087VJ%{M4#o`T3T82$7c`Mk3v`1r-iU!IL6TgbMH7tfBLzmPAo"
        "`bL%lEVKFaxkNQw*}kdDB^c>?ikr<?#joY0!C?Oe{)H#23?`KAftAjA`mOLn*nN_>a<?cJNojx*QqfOG=W&BOD1z%)^~c{}@@mbR"
        "7|%9HV6nX;CBk`>1|~s*pZx?*{`n7Q@2{fEk7uthhg#9`2{E_E=^3i`yG17}16!?Te+W+lx(ttUXh$*h^DMG`{~(4O5UNj}G!sF}"
        "gJ0jF4!>uWk6!s5HlBk!u!?S<`Cs<<9u<?3mn;6j@|~;*bV(dr;x{(MZfpwM*cLG?t2x<8Q&!x@d6tTGD4I%wAnNK?9hjuv`JcdC"
        "l*(yZ;+7Itp?B|RK8~Vl{Og%d{K70dp5(!eOYnew5g*R;5;vN~{)a>bZTRzUsxpiKWaVGpF6&;Cq$=wg*~4Ir`WzFuY7FC#YaZ-t"
        "dV|v&iI<W(u12JL#D8_O$A7!d6LjYT&gi8wtcw=iPYesNypqbB%|kReEy}Qx;*!175B<f#KG~l=JmSNT0Ai^lQUHNE<q~oh^Zm*="
        "$Lf{sMDEb1x-4Ms_&yW^$L~~hX&zeB5Bb1g;w7)qniBDr;B*OoUmlo?2rsLTl(f_#cOUia57nUJK=^R^3n>-kEl-`NojCWfKY2<L"
        "mb0mcjbCyjE~{#+>sp{!4T6I5xPUHf#Re{x$n00+{>u~fQb-^#FH*?$3M1~fb$#`Uk5o6}ON7{{;&eb+2HGHa0@^g9()5?ObsZ`6"
        "#?SqobR_TYMJTXNlkLUaO}>lk3ZJu6Lq15u*8Or}>---8>^dGOTg3_1Me&T8&=$({>bn7?R))HE6g5kCcxsgVh%oik^qI6*<T^zv"
        "You3$E6>WT&goX97NTHNO^54m^pugW8!eR$6E;i(Gp!i_TINjyH4Gg<nubpUG)y2S$PAOH1DeK>a%sjB9ax=(j)aec<yD#ukwtg6"
        "&!t?XS-dRDy2+B-sIKb2j#R<czB8*Ffo7JhxpBpm8nju$rPcypV*62<Vav>1`)-Y3uVXFiYY<a`U}S3)U@NEU^YgVR8p08kr+&pe"
        "1S2a|>C<TygaiO+C*jQJ{Z^74W}EJ*-r`GFmxh1kl*oZ4xg*V^aZlMA-mdoXc;c>f03BKs#%|Ola18C0$7t7ic4F#HPGpqS#kVr0"
        "ge5c7aY!Mt?w6QI_!o_C(o!>@-N)dVf}?6zS2~YW2ln}C6>3mgP^bp05zi8u`2(XFW=X+?@a<q)RjUZ-nS#7FvH9Gy=R3X}wpWIN"
        "0LTZ4N;I+dkff%UdjuS+SJSQ3{EP#DLmC%lktNa@1y}g^+EuVB`7S$hR<*i$Z^GcGHyfOq{gW|P10~<e#H??b^tZt~!zX#9SgKz6"
        "c4fO9b;W+6RO~691a`GUkhsQgp*T=->W?f|;z0`$JM-%=G(mqk^^P@yC0y{}8*v~ojEN?=6VoApxKGUtua}iSRhs+H-X$$>Iw9s^"
        "z+Iu|mNE#U8OBd^`38@~Dfge`akj=H4A5CAUR_ac+FYdBVsOAQi0!ZNDyGCcD}YqQ8rXLQXtv&}856wtZmN&Wu4d>JO^b>jHD2VU"
        "ZtM3H|LoHzQe%OyqwVeV4X9OIk*R_XUV0w@U$}f{1*q8qn31KvOrYsnre6+E6E+>ZNg)va7P8)!U%9%655tdTu-QKZKrne+EL%LJ"
        "sIm>h8jjH}8tDc3>4)@KYcHDhyXlAo^fJ~4iBiwiDNg?gd!6|F%mzlx2ld{?_QV4%z;~Q3({!XF;2LGhbPEsaST+%}_G?=<aj(zr"
        "x&nv!4HOOI{>e`Mq}!?Nb`7-wDg2K)=90EpQVOd#qMLalxt}WR-GI51U0Lt@RuIN2VJ%dyv#}I2hw9b`y6?ekW^aS>y9Rde0QY%k"
        "TU6X6l^C=Q$3Fiaj7?y%2MNQgi0M7z^nGE18rVO^k9|}|X#!dg_)~4GPkryz`!}9At7VaI3kF`mLb<f;p2EWeuYYk2o@nSlwl0j("
        "KQespzcCVOQ}N3ICp6Yn{Oy3i({VC7BmcRc4h3s2Hakxw-DY)$>(e1^gIn``ECzFM9%d@6?6$fc6+-*Y&FwmkKfLFR{tQ|tYB?G{"
        "Q9ox3%oKNpm$K^8EV&%qE0-UVw}a3Sq#FmGO~4&tURR;&Ci*MdS)f?aH9%BKt&fU~vp8Rk0ggtg`D*9@E?P1~jIJHeMe-BE_;A+^"
        "8N2Ki)d!V4f$6%ELf&Hl?yNX`4Us9H3@=i}B5fc8RP>!0Jbu_OH8Og3T|Kn}o_OC;GDGUFd7Ty(2Td0~zez>&j=r2eAk-H}aj5(Z"
        "l)qVzH*rR%m<G329CY&=s_dtZ?B1gkoXi9cwh=pYyL6R?M?eo{mmI9%`H`G5dgwJ9`4U<9tt%z0cLLiU;cRDMP=qq2%zfY4=+M73"
        "lVb-dB{O-0V!z1CJ6rB$eG*4(C|~^j)MO6){^sPF8^iCBiM|Lt2M*CzXg_bvr^h6C=k!1i^6CtSbtG*ywPwd*9TyF&roO8nl{P!{"
        "kW%7tz*?Dsspby5OgC4hsw84+UJPi-l%4s)Pj|FhxPDXl#RcrBpOi*-YCmudPMX|Zc!mtVYeB}Ky_j+UrgR`t+p5J<+isUu9Gy#&"
        "$$UF`9v-{cFDScV+i5SKE<Yr_xZIXt|E2>~;l5Z<{waDz0l(iMlZ(9qMO%-?-O6=M4WYqCk&tYWV<0?I*ujN_(GfaWbn3M*$n<#P"
        "oeW{;U^-r{;ZP&BvP`n%V;5;|Oi#^rTGwuP_dve51dZ1r@vRSD$-zgs%WrWF*qs)e_HDPLt+<>rTI4Y_naLE}Dy}Vj5WAqv_LCbV"
        "urqt%_d+Hs30d0th+Y<xu3@<wPE%?j_`fwL*~>uo$LvZ4{vJG5&eC!Fib(c3Y^2Hqy2WtUIgnmKlWSwH*NU@Lo|xyqhL#^SpiG^X"
        "dhGIEyIR@F-P2x5*xlGsVy<U>ofoQg9)^YVA8hM-+`BQ+dVk18YdbDjllM7Y#g+Yf@J2PBTNa5#9;zLXr{=AO^U)gHL^T#fWPZEh"
        "7lmME5#~jum-V!zG0mP#eV>XRzYL&PiFfs6NIPhWb=>KezyAFTTlF}w>44q$5p?av9u^fuT3cM^PLCy%KwYr0Z=a*Mmg3TWA;>6P"
        "=H)!_pJ472R7RNb6ngI>nV0qkviHP$_T<Ue+C`CeFe-(oZc4l#BhnB!HL882_i_1myEkK(eouFg=xR?lsuwYXsw$Maq3GRsL{)p}"
        "*x!!%&DC@hhw}#Ojy&F!*O)z`R^kuu%{yyCcIsGPtD%lm`Sz*W<^7M>ayZNLPE}c-vr59Tk$;GJ>$DF7c0X(|xd+zGPNzF*ca5h`"
        ")wTg;|D(8&f<h7_n2A!V2LZY#$@l+j$7m>?biClBP+tGOWP<J4?@A}|$>_V?2gKmRr@4&L(yBty!@sTTeyDIZ+T^ioZh^<i3b|ps"
        "=0Gnx+o)q*Z(+H%Yq7Bj@1NW6c{~mh-R5~jod6VtQA802J715yqoaPyx`oMyAd=DTrCAZFnMD0x&3q$%!%bun1@RpOy6%DZF(evR"
        "7%KzSP9h5ygOGKJX}}w?7%PKCY)cFG_Qb3Xl970aCtI&uv3rd7atXUc+r-@!N)verKCh5VI7@#=NtS&{-QLP5x`U991*0jA#r^mT"
        "xckyO={T&^wOLOi&lYz7lmVY_67Jivtlk&uUB4cozm;N~D9|hYjUxbm>@<NqG{8AGwadVDbVfP<f8db8Gu6fLQ382MIZvO^F(PjR"
        "n%Ki-=<!R#RLbZ`&~zk^?^^$A2yqv8HdgI&2`Xd}gKVu@^{;TO7Whlu-aCABVBp_Sxcu<x;`Lc{arWc+<@tyAm;W!N`k%VX5A||v"
        "-a*)%4qUKG>`K{W%{Z*Q??R;B0Z_l9322%SffEw<$;rv_O^By)axD)(l-SZ@G0Jcuzm7N_l&s@Pxu7p6SX}*UtBxdTM?3K@eselB"
        ">eT4W#Je1dQG|^et}Lcqlap)XJxF%5-M~r_xjS7^(OSb*IMvG9(IH<gbTGb}=uflETPTm+M~k}Es-d(ldUooBVZAV*8}MVO8vIxz"
        ">*ovf-gG4~ptUW1F<3=w>W?Ak<ggtMTeA$C2(yk}E6gX-r5=)By4n0cP)h>@6aWAK2mk;8Aprjuz0CIr005>O001EX0047xV=rQ9"
        "W^ZF*Y-eR{ZDB8DVRT_Hb9ZHPZE199E^v9xSzB-2HWGd};QzqYzDVnH<hFZ3;o@v}7fHGW7TqFo+lRzGpe4$OjYO&>75g;Ef8UuQ"
        "B~h0gCtVb%9}<g~;czbBd_yNm^5m1huv!<qW`$PH@WQb-yp|<*QmZeR(vETV;)kbO+emj{rM_3S=A|%98|H*{?7)o@*5>DD=V#->"
        "mb>D>q-D3=-BzlyE2O&1+4TX2N>R&OVYn0Zkr|L-WZ^^!hnWV>{sn{|xGHO5&(FTQezyIbHM*3QEJ!KKbj8Gjv<@V)O2V=YgXQdv"
        "FtA9#k$O;|ltvVO*FCEBm2{wJk|e0DY+7v`tf?wcX!t&Gdx*b(vRVxvMhw5&?zT0$fV&SvPBblQbbeMD-7v5sFKTWrDAga<4}~S5"
        "CLY#Le+!?e8qM9x3O#ctnuh1Sdj?6*Ma!jm`oNpE790H3Ryu8t&*b#`h2pI}Xoqso&q`4-8%Fkc)wKV*TG^Aa?cW(V=<588!B5MN"
        "=zX?hFViYV-8ZZv)r5kz6~sivq+%8n6y+c}4_<mLj=P#Sw<TxQ74rhJ8!uW`tQIPL+me)a!}cgB9QZC{f7sD{7|4P`_+z-Vf_=~H"
        "PCPY6n>2aSwKYbI7;Z{-wt15IZRiEK1bQ|vyQa0h+zk`Tc1G;EEu`E%<Fy4_S}-+)t2TC*CYxl#lB*=ka-j-ciZtn5wf(#oZG`KL"
        ">Vqh^2maY#|CZu`S$+@?rM!cHPc!d~e+=P}g2#UmYIkiq0h2U_dG`Xd>A5JhDZ`bevW}}l>}7d{t{HD+h1(nPMp}sXXbLj$?g$gr"
        "Vr+;5uVplUDkg#&)^=hlaZfoH#y;SB-}L!ju3l95(7u{R`fqaGN<N74#Qbo%<&+`i-Zwj9DGfr*C4QyRohkwN+PzlwaSu4V*A+mV"
        "<n<n5)h1)Q?(d}A^R}&zsmC*L#uY)uhK(lQus>};8$dw|HoMijhSNd-7Y9+io(}dIRA@jKMJb)YIT+mA0{;b1h@5v$7pAUu74E@d"
        "(t}oQ?++4GZFwPfO5Z~nNXA_56&ces9F&6qp~@8Yhc$iXz%H-nsXeq}%r)E94pbkZI5(Y27XsJ?+-{rPY#@8u&h5x*^ZW%f-Iu$;"
        "!tlvkhghIHhnp~VH_XXKz`*X$pZHU<=*p-~?g0QLpTwwdsXqCd<qI3DIKU(%bpq-JVb(#ZHF+swO~LpObv4)>P-qO3p?vP`y`U-#"
        "S%CSJ`v0zGctQbV0Vy5D4?+DSDG7!1xpx|-M5wW#Iq!>a8OQ`B+<+orwS!Q8%`s#Q@dcFmL_>8%o*`8Ql^+t<GAe&P(6O(dK<y9@"
        "fJ1@_$fypXn8*i=!*Ak26diy)+9>xzzXKF+TdG1(Min>5OJ0_jz&wN#mqsYw2(LaIj&=tRfsP;*m<O}q2jzgA3?|Ko`w#9$mAzr}"
        "IU1o-(e!{wvvGpZ9O+M+K2p&90jxiO%mF1%x?pJ(j&V*R88-q(e(oTY%RGrAl+GQ>@+}D6V~T(rsIPN%PpZ^c0~QNuIv|9y1mmHj"
        "!Jv_PP|Y2(zbV^{)tl0inX=hJ!#P}^()M4Ic%<sg8fhECfk=g^*F36&hq!ST`D2<OS!CNSASFvq`|RmMD~Q%|WSy*2K0i>vaN=+U"
        "B1iyaZ#^lx8d=pys^;FKJ@0~`p5>yAa!TsdB<Yt;mVUYWF;Y59Hj4`eWeN!WJ3*Bo;5Z%~B%=awpc5wjMl0e({>0b_2SJ?y=|H9h"
        "9ip@OW%}~RSN7&2W4FVvuT(M!%SKVb7eONE;eUJi=?%s~`1tH5I(*#=|Ap!bw*R@)=!(8=*rM%bT?Ho12pSZn^>OeN{-9^rZq#I)"
        "nUe3+6XPVA-iNCr%osO`Cpf?pUKm0fV^yd9S&l;!^RC`-I!z$;Z&^YLvPL*gM3=ZP-49I~0i#b*pO8R-u-Gx!(t?-sWGZ-i6|#iK"
        "?0F2+U0@bwLO?z0BVP(>*Aze?(8f-0NM7<Cki!BIT0?=zXNInlS8BW6LcG02q3>R)i;D{gx&A8|RQF-`sDl5TmySzcQ-%r>z!p*+"
        "XbAMWlre)PmtinHu$D^1L6Ii9E-^X<!&<Mz{rC||mnUfLk$(B=>+4YDh!4b~F|<&8bX$)OIB3&o^`n92Q^|=3_tA+H0->jI6y6ia"
        "T}H|Zo;Ixu79WIDPv1A)`L)2#7|igb=WuBYl}A48lDL)Ll8h}R9=Eia_C#k76cCb~@^M%c-RY0)2Ltq%A$<=-i1O?UrjQ}Adz!Vn"
        "Gr(YowE}C5*$VL&029-bi{4+S(+9%jP9WzUwY;P)Ys(NrAeZvyHLOnkhYduQ5^Jwt_dVIlZo0!BU7k*}{3RZEgUvE<E=bPZz)T@n"
        "F|>9Q@LOwOmSt$@UV?&Tw7GV_MpU%jH+p@wEoCvkS+v&k)TkVOn1d)L)46xDTtn=qCk=-22y9UB=4y%Rk>=j|*!Iw3=s0+p&O8_r"
        "Yt3!Y_mrXf02XFFDP0^eTeqUqGD-gv*OCzz!zH%>hYbJ~;)H+L3;98(P9S4+K#v;LluCFok5PSnHv}*4UMe4+{qjJ}_V~TB41ZRw"
        "7{TakAy7r1S=nY?$kM(aEdjbB+`p@m{{G2(mO334-kw-EGuR^ZeJ<gfMUwxdrAqzq1O&hLJst9hJE7=Nx9l^AN*X=VVmpURpk{ZA"
        "?(n1FZzZvQYm^#I@sXG`Zp?lM0D=uxj2FnG7csfAkK$pG=>AO(lBgVdCTQ0UP#<hfed9yYM>qePitvBbfZwal*JAX8%?^F+Wymg{"
        "oWncfvYqwj->tmhrg5fEn@SL_BdAEA(~)mPxg}vSh`6qWPG`08nfyIqG8Fx7APT)4^0xspn)SdpeM*1dn^bg*$qR8huLJ(EWqdZD"
        "L1BV_wYmi-A1;T#U(Ml?pn9_^czEPmKNcTk7E=+VL=1ehaLfE*X}Mx}ta;zm?3E;1Hhpd(f5a-x{(`Z16`8%BPX4-aO3Zlo4-o;O"
        "@IO#X0|XQR000O8001EX&twv_wFdwI)fE5$B>(^bb8=%ZVrgb?V_|G(Wo>O?FJxhKVJ~!VYh`X}dSP^FZ*DGdd8JrObKJNQzN<?A"
        "0SjL^8;X*=o6TnIan(NTln-&XY*j9@iiIIT4J;C10ML@+=)b2M4-({zq)J5>i9~m!A76jn6s7d{vme&vE7zhCye6%xSWB$B;ZppY"
        "Yr>4-*5pJrji>}1y}uxqcYVt&&q>d0wL4L=y(gCGYmh7`rN>9dN22SMwq(c5t`*zy`HNBV_^8pUBmNB>7MdG1=!%>0jx?g>CWyfm"
        "Xdxc`0gq_komtJKX_W4`9(8{ith$KqkCI!_U|`q0>sV1K*@)|?@Y{}8H@y<le&kI~-rn_)i;heCspr)r>Tq#o$16Ybo(cW-jzL-="
        "De?At_8wh1KB{>`&|zjvE-O`YU7|UDXEW~Q)^l?!a`J{4tJlXz1b#G^paXT<5c*kumbCZ+o@f3Le;WGY#S8LIprLP=#{v`h@LBo?"
        "cbT*gG|^DnC7cLJ41~k$=t^NE>&;Ht!WASZH~fCnvTj>5r?DpE1uJiF6}_~Bmcf_z#ta<pk46PPKcMs9Vl)qQ%1*0TT}C*TBM`G{"
        "EYo^fvQen47>KPMNLC?ZfXCnPEzwF@g7GsU4bklO^jfLwmY1a~>&cHNMNz!+op4XK_ZIND<&Yijd=m1yKp)ZONB&h9p(GI?q6S%m"
        "3&c6d>k~+$YZ`}r%$kIIg>1OSx;Z%2Li5TheNQCoc;StH;PzJO8`5y-DY%U=u&a5eEKm9oF+&*mBsq9WxMbUw*TopC|D$74G~8H#"
        "33gn80hkmeSkH{kVC_4dt{kGUrWu_Q8x0m{4rDZkDtXu<jKSRv9c***Go2%WLpN2{;We7LoGMv#5KUW(`pQd-W>mPOIwYB1k?%Gn"
        "QHlraDIU#);pCsJ9r#<VmCoo$0YoRo!cz1=H_!qIcMGs!LoLw3F~S(ZRHV%kVdRuh01$=W2Z8|yfgUTK(NmfeT9(lwb{ZBu2iw<J"
        "9dKca*VTkx72r*0vQ-it76Y7^96EJoCIQHsiwxwTy-Kt+GcctHiQ$8??+~;;Qw{w)bR^;pD83bvFUcQx#8F6$B3u}ly)ZlW^3OkI"
        "?rw()*E!j0RqXhkk270g1^wR#y-Et#B$dM}NGD8(Q)cPL<mG10`31IE2~`G?hH&Zx_G1QfKj&PgOo{VrbD;+)jMam&rSG~Ddu{WM"
        "wT2&Jx6y%oEaJps@eoWQnU{h5ve*@es6O|H25hYtV<=z5oDy9)`#z7Ye>aX6CvVn(Gh~7v&;I&TB2@DygxuIUJjf<qjpyXkyNei;"
        "c|}^$2^fx=e}SYB1vJ&P>K5!93-|z-5d&$L7MBJ+$couHx6byD_rDuYcaFR}fPetPBew!t0^3)@i?4uLA@9josOffS8FUC-a{lt?"
        "K#Lb{H92WC=j578u0aR*0IKXp!TYmwPtuIkL)W2K@E$wD?=)lwGiu4Oq2<oV>jN1luQ%lUhx4<)yd>E2iQGKtlic%M7Sq}RKw&Pq"
        "3;=(e7lpSl@>cLHj$0o?={B|K&7rdbmJHF)Xk*_KEOe+%%;CQVXey`?7?!XX+PDe&-Ed-4A9x=BD3cNO*vQ9DC`QEv_=DZBJ%N>{"
        "*Qz*=!l|eYsAP+`iBBIOnlYcwksy@hC%nlI=|36MsCYQ0eMjwp<PqKl^(;h~6IRzH+J*Hz0L!u2YEIaR#mh4{Q=`4DGF}hPp5tfw"
        "OPH5iX3|kni#kr%9=1C2>pwmf>BQ)PlzUeu`qA^J`)Xihkp9D?77ftmj^Xxy2Y>=}!-9>wA{pFx!fp>9$Ep{5Yh~z84xGLIzb@zg"
        "?!xV<;kszO<Hj%;W?=C08`Y}ibWP}BxW;)OR&03XuYCy14P*w4;~e*hE?D2U_hkjQmhp=!nN{(U2_$|L`EpM*R{C4Q{PO!?iPUg|"
        "OWFSPLK1B`?3DkjBjitMw6GcoV;RIb3;7VUJY40IwM8(~!QrTuQU~M9X8{KOi*W+4F>MXfW#&upG{q&~x1?4)*%>v?ncZxJc!ML*"
        "{o|}>V~#`5dWnNA>q3b=`+{7+O%(%D74CyPx&*a?_Hum7Z{51a8VenPj#aJlodO6i$lpdcqe){`51Q5qPdm&jh|@*s<{nsfZxg$y"
        "*P~(gtpnpV48vw<+oysZOQ6GSc|L9^UbK3-Ca@)BQ%+Zow5I1L%lPXbZ$liFR<OqUp7zS8VM2}Jj60=Atq*DRW8no^U(JA)D8YE0"
        "o-R+$ox?FH&?GPqYzP~Ep-@I&V<G<YhCYi*fMBmAymuQw0~-o#$Dx=LP*M!|1f+|>6h;^U(YZ%Pz|-yF95Dm3$BggcBFtGba8~5a"
        "p*0Eeg@Q5DQEIW7KEE1mhZ^h2?<&q~UcZ96;QwY?Q3*TlnEne8wrckeqk8l<tn-JL-Gm3nK_ij(Dh?jch-nNqPX|xTcS?aHMDmnn"
        "EvuPjn{V?0OrzoOdA{CTHu8l+%-*Q5+_;ts3r+=%CF?pdt5j<qJ@iTD^<x5IgqFR6s@%H?(#G#E|6eEgaKw4Bqx=4)aG>iwa6OiR"
        "nZgLNrF5v#OgL%xy-0oaX5v#t_r&*64?6oV52@{ivHvnrdzhx+e*sWS0|XQR000O8001EXr3N>f=>Px#+W`OoCjbBdb8=%ZVrgb?"
        "V_|G(Wo>O?FJ*RNY;|FDX>V>XUtei%X>?y-E^v8`kIPB}F%*XP1M?1ttQ4d#ATF#PrO=Ktbs>m^OnRmUbK^`ZeS8x~3)T^uRmgwI"
        "_X{Cf>WRr~f<Xr+V2Xj0@muHv!`$>iD|=7QsY~dx@69QG2ilX~B@;ZD0enbcT(soKC)&(}5Oq~oJqHKfSg<VxGx|_=r%LSWDs|XU"
        "N#+mXtgav*e161svBKs_t{!f2xxN_RTfD;-n;Vq#>oqPH4gM&RjZeACMa5<aUMAI9`X%N~%J4}Z+DBqjh8#TSWsH={S}EZQUdEL8"
        "Ln^-e#02X>o2EUI|KXi>9TOMVYh8T;P)h>@6aWAK2mk;8Apl&&;2mKH008tC001Ze0047xV=rQ9W^ZF*Y-eR{ZDB8Ec42IFVRUJ4"
        "ZZBnaVQgPxZ)9aIaCy~O`)}hm68=3v{)d5zB6i`&X5kh+=Umhe?bd)_u)W&@4uwEVv`s`3U6OL*=JLPakdpOucC(8f+Dj0`7KcO5"
        "H#6Ty%Cf8_J#e8wMu&sXs0RbK<u38x(;y%qT`s5u;`4APnMXm=0}(_~{{)gp!o>-KZmFQ5OJ(c&>iTNEXA%PLMLq?O<K&PlppVpz"
        "6op@+fQ0su_zjp27K;Yx&kH}24d4Pj9k^tQi<ACHm;goOy>`2ZHa?SkT<yFgLBZvrqK^uA=%4P)H^F@$ZCoDUE}7(^L`T0z42=a9"
        "Q5dptcYU=XlKLz}FAq%ZLG3Xt65wtLE=N@Z_C(+)?g<|wDB-RFB=kV>dm6SC=RRyAZ$}kw%(5&U^j7e|%y)evC3<FoK{b7pn5feM"
        "-JM702gP_u{M5rH#(t-<Z(%@NmfG6?TG-3+)Sj)KS$}G;^qlLfpK}YCe=tabTN#|X*H?f)mQHh$r3Aq{nDLN8Kna2rh2%(>Pd1n{"
        "%!AI@kqWle^<Xwel5Dq<c(qf^G4-7gk?Vp~B7V5n38t7-=YYGp?WQsQd+T$eeuJJ_H}SAJ9A0DRvm<qJwGHln?;L;lw6%PoPHgEj"
        "MPF-sj`!2{uHaJYt>_Rx<Jth`L?Um8IAvD)_q{!x_B+mWKC&myu)iGPsI^z>AiM20VXYnKJD;{ZdTg7jXnWk1ZJoh(jK52S=X5YB"
        "t6%MbJsbYwOzi&1UJm9ZXAch#EezL0i?tQBan5i#U#*6?w*9d`ahCRCo}aP9?ua(VmIUMv9yX=RbP2UxSI2KE;TxD^`(@>l>rWQC"
        "e&kW_4(lr&KQ|wQQLv#Rw$ymEXYKy=)zW@vFEI?;8T427WIVH1UGSKzG<1zpb!#Y}k_tC`)$BCww}a+*ZFNhgwb<;EX`@nKy-ug~"
        "rjg5Fuac(NUA<9Wz4Q9jJ1?#N{oB{C{v4OlH!vh24;hjNGTn|LSV+*wV~DsRxtL5Mq;4$4n9Y#(hLbUJSdLoE5-WuDGTnG|3(l5{"
        "fGEc#x`t&CReP)s&7U+UB=jSwibMz&X4nF@1v3dyCJY5pUsBjwPqC}k?e^KqRyzn5Dkmd!M`@Bu@I-_hL^tx4%oj<#hsH&g()FEJ"
        "I8y}pXC>y}CI6X^qNFrjN1V9PoZo|Y1h<L?COp0Oj9wRdpE)Ts5T}}Yrd+a-6wLyv1al>BSsr|H0)j}i`>ivvuBa-k-&+fW3t>kY"
        ")q<4u4vz0Bs^p%8n#pA(j2;_AT8T5<1L{Zm9Pa5UZn~7J8-|EujVl*pD~&Mz@W4E*rXKv`$N0O+PtEFC@LfDsPbiygsD{iMPbME$"
        ">t%mE{%8Yo&C*^$Z8l%S)b6i7EbUR<ltez>xFW38hs9ztwnxb<E5G2@0(dfOJa^$K%ki^SFYF5-)4hY>io4v$(+Ohku`LeVA$C}Y"
        "rl@kK6{#QKc|&IqiHi%SPOXybO33G(J>5+8er~$|eEZ`w{0E+H(o+xAwA%fATi1hc;z4~5%0IvTUO%4<2XOoWYTwpN%Qo}N#v{5i"
        "bnp#Bu1LFZfkj)Wg@LTqyBQV59<%Spm-S?N<-GS|w`>2clEl}~p;#497ba(tUq6R>$*f$4_2-=TH1JE1TnY@glB0pkq2;)wg-5&E"
        "O&aD+{ZfdP@G1^;sAg^)<`>N^$B%x_2)wci<MVq5Cg^F1o(9^JE~%8GuyHEFV*01Zz41x>|A>Ok#i4R>A^EF|VUFaY;8P=s9wWPu"
        "OU2xbhg0g*oY^fMEMtbRoF5If;?mqdYR<*=<lTj{qgKD%>PYM^>H<;jzF{bcq=7_u*pi0o>EEA>p@xUZ{&)#(7|y4QiM<}rXD}M~"
        "-_Pc&^>|qSo3VAFCo8;-N9v}@E_M+n9LYGx+u7hpDq4lXq0Y2!I>Y())X?~)?)~e@yjas;#^u%GPv>7UFVz2EK<LF#zn0ajL4U2M"
        "0v=yZ@LB!k8U4~Nu+(N-t5}~GNf2FCFNZIy1AWw;7vSNxpbkmA&gSHpeB~vgEEd20DfF*zHCy(m-&-Ky%_B+{2~}G?@Me1RJTt3`"
        "yi-c{ruQX8*mK&4n_j2JBdNQva!G2zM}1+5rxa_e@zTqM_>5f?f?8Pynnnfy0d)~ivZM8c_xpw0Lwx_wE)jTxxGP*hP=_=<%4mZL"
        "Rgp57i1R%`=Sr3ZB?|9D{t!Be>mF*Rs+E(7*qi4SM3O2)0V(>)v6LVT(K5i<LU4h_m;lcsWpvuSjbo2ZGd0c;)}k~zTr2vmr-l7L"
        "7Pz<xZ{KuEeU!U{yqkYmg68KYb|0dl6>z+6j_ml0*2ER`XCqkuI>!M>sk|bAQ=NUxRVIM(iNd_b^b!da@k`sv`&CnJ0TK7IB!FKn"
        "bVW-0^HeWG(-dXO<`;V{S3&2ybFV@!hE+314S0I4e^wtqD<jEzua}UE<o|+~PfN(>J-s9$FUUxY`F{aWO9KQH000080000X05_!f"
        "&KeW|0M|zV03-ka0CRF<FJftCZ)0I>XJu_|VJ~HNVQh6_bZKvHFK2UYIBPC&d9^%ibK5wQ-&2+Uz~g;5Dxnq0Z)>92n#ws_xt%<2"
        "oY}f6N5!H@NaC3yHGKFHm;d{9HvpnRQnupUdTTc#(P%XK-3@SE*Zq=YnTVagUwl0MhqEcfCM?1#$ulR+Vy6(P2urb7R(Gjz$|x_A"
        "Z0S3tgc8o8$kz_eNebuU{nw%Ex-Vb6e37g-c~LoGu>`nMG%r?Rxk{6H^ZdVMo;42zz!APEPg4;g4W*;T-sN>xi9!Kv!V1yo!7uQL"
        "tNwBR&kq+LK7Vqij(ad3jiZzJXmqlFI6f8or}MMX=rlY&J~*Do;&eY8jgC*x_9Gzo;`5(>ygLuxfB60U0;b--eBr>KTNPoFxf928"
        "$0v(}!;?5V8XumVL`S3eP%P%B2WLlT@q7+1j`z<_!g+XldMu8P4iAOc4?__jjP?&t#|MYUN3QRj9G)Bue2rO@6(K%2IT*zU<Nd|4"
        "I5<2U9UYy8kvKd(Js69l@#y$CjNm^f2PZHd&P8-~HjkqDnK+yuL<eW__z)0}55{MR_~GS?51%f+{_*F#uaX0vTI?cEuZ2^sgtG=V"
        "fq?UQD%MU}iH$SAqko6aIf(iuf#H9LKW72_&u>5{U>qkdk}OP}tX_*Ei9jvN8&M2_aRX%f%NMa&IN}!A6;*-Kmp}q}QNG@!0wp<^"
        "D0*D?V^~EiLad<wql6`jvXZOIsib`{eH{YSI=>FnUz{v&h!F5XDjOnQ6UHe(4oeXh(aI}a!sUJ2``hJc@9f*~n?HL!0Dduw<lI73"
        "b&*LF!)1}z8*e-qmQ|5#yulDuI|Z3|t`Bovci<!owVFsv;e5(7VVH0Yq%9H=2W4ItkqC<Xru5>Z0A0?DJFuEfnpBz*;&ub?5qK5)"
        "<|#r&hplj>yrhXUIWG>Y5XFm1Z;%h1H;zYO&TGdVqFtr#0O=3oV15T0=c)a|29y;JVded!V-GP7hAVL!Cri*wZ{Ymf)FMw&Oxy|+"
        "*!Dlev=-+@kr$r3EZ3)3ddG-4SSKa1XyO7ba?gs@zz-n(5~&-ec^H>onq*=C3RgJz%SkfY-;ieE8z37g#-Fo>{#0Zh;<5=P&h1Vr"
        "fj>g4R0~|j2zVg?G~j5Me3^!S-c)<A6#{qi1^B6TU4>wG&ZeHHpuYcg@%hv5zet6-BYB9@nG4{9aKPqQ;AD+<D=$|UDGIF+f7Ku*"
        "B#G2PFAt1usYDB4xW0nl9ynx?Rps<+Q47?&q^yGciXPPrlm@ZFcRX85g8@p2tL8AN#2SRe_#Ojw;4mnJOc{NB3TkG7SlPrO(PiVs"
        "zuhd%o)hI6psj^@3!DVX7VfI5Hs$2an<Bg!F2VcN^STrT!UKnf;3?kZ8<B;{n`IHk5CfxCo~L{0M&I1)So0tU134JbAgb`2d!pq*"
        "?bIwzZq}CN5FT+Q5B~L?Dp`xXuBOMMffJUXh~)+Xp=fbJxdA;&hVGpPqtOWdJGL2%_CiMP{OV|skpoix2}kglg~j<zhUP8ZL0$x9"
        "@?CIYb4lM6phC(;5dp^B*5MV1SU}PNo&tj=$s0I?ky-tI1wvotpwA;_e_v{_qc9FPkTQ@u<K*%e41RG<iG`KuD6Iku%t1C%wICv;"
        "qGU$6kn2DYs?Z3$k?))ikcpmrkMRl_ZHOB<Z{?i9E?x-%gLMs?oQvSQC~{<+NS3Qgb57ib7`xDgy5F*I8CX%CWS+t8{x@n9>}cYk"
        "I|CXn-M`c;cZPQNE&Gk4s{(*<S<fL>zI{7<^|xPtr4Z-Kc`^#SGbS};B#hzi%_c7=M7eU#I>uz1x&q-W^E5_>Kl9<g#<}E+MJXy@"
        "BNG(VA)7QZ<{Z;I@O5S|#fihEs64Z~fp1<a(rp_cv6V?yR$&J5+#rQWPhvy>{Z@y6{hm#f$mO&h%>iNz%wGl?I|{Z-V<8p-JTqwC"
        "bXzth|FQRFrX?k%BRdk6WRO86anQ`~At{N$529rh<SuFA1Z6b2mm-{u4&w*r+oarw;Rc*p?Ae6e+mEELnyWz)Lq2Sl^Lt>6JQV=s"
        "V&>=e0nA|+t94jhAr#Tt3)ZO2#AwzQ(hBJ-0FtHFeZUklZYig4jP$iKQWR42l9EQ(9dvQLwyd{rTvOXSw=wMYo*&F3TNMpci|Ky5"
        "7#wX&6QgGjc7nuB-ERjJz3(AR%wypVtmtAKy?TRXoi-|kzJ-u790Id4+bw9JZdr<A10`$aU84m&j@+<^64ba}Z%R22%pEeY;7Z)-"
        "yh#<nc~MSPZ7{)fTklhO!+x@4=n*oH^i(Fpk{Ag7z|+3aCP#FpC!b7OFQ5BTC+P&<Lqt2LLFEbFLCON}LnjkAs{U6in55C|ap^v8"
        "VUgToN&v)IVcIO(o2zwqE34sw|BQ)Lp!1oMUG%@c6_6Bbx*0G_c`@K;;ad2zPQe)Oxyqx-B+ge{fMDaft0V?P3CNQ4Fg^@8<CTmc"
        "_tn@3D&awY(ejY43g}GxfD*{jvl8-|AS|OKnZ6Ifg$$(0yF4UQ*hFB^K+yZCTUU#{Q@3OO>o7|ez;e^+ptS)L<gVyC=gA_E!gML1"
        "_ExnH`rP1p%zcZig(I#ZI+ByrDRB=JB~t)$4<<B$YOd4<Njrd-zC+L3%8Lh%b6X^EN$@bj6@Ug}vCh$-Q@R%z<prw;aP!RPXs3t>"
        "FDWJBZgR$1FSTOEXEz&0nNPHXYqWthfl%Nt(zUj#^(dIl21sVBJjzpCrKL!_1}_&>5~|Xmf`ao3lm>uFzuI7+4pE(=yW0j)LGe>="
        "8;ph2Neaw`5b<3*O1WzR8CQe=I_D|GQh-n@Y&ZsC6xEOsb>~sWD1%f9IZZ%J?4d(*?!h*KOP9q65bD&585&AijuD@$M@TTPWz)=h"
        "H5B>{464(8ZMa=KZFi&OOcOMyo{|xR`M_6*I!bFOKv7SWchFHT&#P@+0p<l*H^!qWJ^Q0zOWW(vt>Xh_4HdQ_q>)Q|?wWUlTA4OR"
        "E`Y#N(-Ou*6W5v<l-(;wMYZW;zRDq&ZiLxRYpXjIArMh(2}Z^)wVR#|+E<AssKT8)*g}2VYtrgX-x^4^!O7Lht&+Ffy}boBhMcVR"
        "s;twBYO!v_+u8r%ez?BLe6~H6vn*+nJEdAp9OP`jRQ_^ysfTAfmN5u#*iK!%W(CG8)p)dVw%d^QDDO#jh=0r=tmdVSQ*|=U%UNHs"
        "MWj9+ad1O3i_3hv#aR=_W~g7-qJYZHWAFo#1rG3+y9%Jlzk`-&n`%|vjfPF9j;$rsFA1}|Ru0q@@omkw!SaIUqTi6*(2ho`vZ`7z"
        "M+^%<7i_S{i_c}A*0N#HBW}iAmNxCk2(^-qL1=(CMGmoxn;{_}Q`FTLrEQk)_g+Z|-(m4bhXFi+)0IOfC65z%Xr@X2A(c`|Oyjqh"
        "V5kLRaG8N0(I;zGOIx2H4L7J8OuKkM;!#^88GnEol!9`W=@#1tijOsx2N=kREc4E$iX#MmaX6(#CWIj~7;bFDj&}(x$t{#YW@5nB"
        ")ccy>!B&yj2jkeg6i)gajqIeg7{A4<=)NbHoFzycBwBW~mpZ%3@*8S~#Wi@$D1;2kP|Pic&|aFQW|D9TK<79pEp<ffSrM;ng*;3;"
        "eBLz6obSn|WlETIMwP4&w=oV8ZRMQWhCsOx@#NFYe73iyrzwvT8Sf2}{>~n(T-DSJ)(u9+Jp(FIj!R`BG!^~f=YJ?C3qQ8vUy4kq"
        "+DV`YE$p3>33I{<b6X3Wv33uv+oCnFz1`NwbjwU_I_lJ)YpvPv{^`@}Ev)gnG`YI7s)A4#0=BAFR)Xz<sJsds8K&(nn=+jm?KNF`"
        ")mVhwOwDK!++}u0Bk~PwKcsY@qBXdzWwi>@WSvxP)sSLFySwDdVNIDiucyw~%v11^b?rfAtUI+NmGVxM6yOcrF=-e+QLJh;DtWWx"
        "8Pz6KxD_VgGOnpM5F6HvpEQioZ>+vRgSrO}63ng05!A&#LR5M3$pTRPqdqPDGUcw$9HeDz3eaiDQXuCoHzJ~5mnue)xm2m+k5`y|"
        "+y-hOsPZe3mDByv;i+}ZMu*w@UI`#}Aho;urRm@~G%xv%!6GaK6|wL(N6)M7mK1fC;cXIj&W3(q?7W@!P=Ig82jkJn{tnSo8=^)r"
        "uVc`4R+?>1Hg$e!*7%uBbN9?;ryLTtgPzB10|lu%&xKujsh2-^G^E3c#d;JX_bQT#3L>}g6rk>YQ%<3eL}tDd#&Li&CQz6wMWWE!"
        "J){N%m>PBEz9Z#&QVt5ZNb?&`uu@Q#^5_nkIvM@5l-H!ndIRYQsHx_j+R9LAXe;DBi5L5{>M*upiuOzy;HNA>86G;o^e%{2VFeKj"
        "q7R;mnzb(Po8~9pR6sRTw-u<3iG)0GccUdYZMVZvB+PNksq$!pi`Z%oT1ZS%y-02wMbNv@ilFw_s{*oBHL$E+p#uYxzb;2JG)j$j"
        "o|kgAon>y*1_a;a;GynX-8CRjA7L>6DA}QlE7c~0byKxv@|TKSnJ&tQ#ykSNxuY5M%GZsy+aMD++=H@_9{<WlDdYrd_ZsGKbao6d"
        "%nkotQ7$iIsvAa<MCFgOTpq=Pd#SOGKn^2PNPI*@UdlLNLc;rML%NZrb}27Rs2fStlWb0F8a`~plVmDy85-5osAOAM?N#4x!tNOg"
        "V2G?oMZkY|DVuN->05KZOKU(A-~ucH`{}31IBX!Pf7%lA!1!Wv5mqMnDB32}_M=6OZ@F}S(~y~7r*xli%WV`KNb5xYRx<lo9?+wt"
        "-B+wxVt#6SLhoitx3xSsF;2hzMsB9{vSnNLV-IYvJ#8ECdv^uw!~N*J4YfOBJfN~!-UotzFZaeXN)^VwGaeoL&e5n{$w-_CI{{!b"
        "c9#gS*}k(KlNopG_s$WIX-iC`UeO_E3yKwHlg8gFc>!fmfn6n=eiz<G)#T?`Q+5YUoiSm2WbgT@O^<tqgwX>(R`TnXMY5zT$|*V<"
        "2Mj5?+T=;b{a;vDId}r_j5l<ts(hnwsZk$8y2mIIW!Z9t(m~-h0jL^6x^}007MOTR_c4^Wl5eZLh*tVt*9k(rUL?!L?LoF@1hj1q"
        "nMgSV8=xa^><Gx|ZKp%~5Ab5$wFugVMG)np3e?58G5{A6G)~zBb`^lsZZV@>>0=~FZix9qb@zboa<&eO8sE(Tn@b6F#KAHD+;s)h"
        "@t|)4n%fF82>Se(iyfd5Vd@q}JTTFnESn2NLYn8nHVOTBh<tW9B9hv1<ENuhFV@|Ik(RdYf}OWe$EhoHRL~KOcC*wbcveBjGFQ8%"
        "<GoMP0ERU9*F{9BB^ldRf9#M&j-&!jr;la7_AqJJYqx^exuC`oL|u@%yjQNWB2))Fir44B<Rq|mxQ29J-ag#oN*I$QgDgUg(^UhS"
        "A7-FeW0W&uV0LY#$hvgVa~U8ytAE0y^(GqIR_#cP{P-}c<8WAr5-WQVLTV}2d2ttDjB8H!xSx}Ei*~+hufi@RNhA4+?uT=fbTiz&"
        "?YlF8_rq6^Xr#O#<p(EyHI2=W9?frMUecAeI;TDE;_>9HN7s8MJZV_FW_4X7;`v<LpfHJQ4)NPJLblu$u~A4V$a^3^aDi*ki)E44"
        "<*I#lY*fW=osx{`h67vn-fh6C)2M#?b{kWQcJMIp9a@rWCPYV!U-l(2liI!e{PD~0=U>nN)9R5@XVO}b4PuN$JK|iNY~wDU_+$p_"
        "8b2ZXAnJjG*s6z3;zt$IwX{xyZ&O9r(liadP5rQUL*1Rc0L3NS>AA7wveP?rS4~}};5nEz2YpnSqVvX2Ht2dR%woO_cdVRP1Lt*d"
        "pvx<+pG2X-Yz*2-M&P<oSsIV*gbVy4gFFv;Z=Z4l%4$LSj+|(z6r}t3K}^pjH=>A+0SZr27X&IL4}wlr0&8R?-W>u6n4|zb`+aBD"
        "a#P!T$G{qk>ja7|R6=Xt?K@^xn%jRvOj_BF_ruI*F-g|s%XIdEDWCq<cKDlW@_eBYFBWA~BpcMDX%NJD1k6z>Z#~BoIszL~hpwK>"
        "rxIsXQ<uKK%wqf|MEq4Jppe`}s0<o#z(hhIXM3(1acw{X(Jd&?-P@B_mG+vuX|PUpw-Hk;H6C5EgGrjg^XJQ(qiQ<kFwrUpA(vD6"
        "oe$=Ji?B|s>F6;`OmFx0=<7eg8Ilc8vGj!?PysRUawXEu)O}AM2yWyD*!<H#X&2H?RK2R(&sunR$)Sh!^xow_oz3dX=V^9ro!?wS"
        "cqWQn<@HqkeEvUQ&j0xE@%+=*;L~p(&o6Xcr4yeX2I(QqBmMjnY}$%~MJ}UJZJK`&r-~|jEEoO>0<*`EJvCcp(a)USq>p<t1U`Qk"
        "yK2*uY>?rby8&(Xbn>~QWVU0$-u+WpAiC~BO0uUCl_!Y7M8G(4`UkP`ue~W(P@bGl_O*Am&k6~Qd{!$v?)&JOfZLyBu!NQMUF@%2"
        "dMEn4-q9buNJ#p8k{5*bRbKpK;nW_$zn-to$UQg4Ei>WX>yqtkpqZ?{GcY7>gZ$B4SN}cOuNVh(t0(P~Y~$OQ2)m6_jlnv`+|sY>"
        "Z9VMLnUnF!YRb_v)<vrw+H<W7?#!C;&%V_6?EAfXMUCrUalLmzHwv072fgKH=TbDUMEfuH?Rq`cObPUdBb~!Bny>4=-ENw;C;m(w"
        "tugczJ~R4%5-_DtKM6n4h-pv%e@NGjj<$>QE2ux@H4-0w;I$IMYkd(xmr?Y!o`-=-|6wGc!XpUC;DZ2*fgsQZFH3`$Fa8fuO9KQH"
        "000080000X0Q>}I39bPE0Qv#|044wc0CRF<FJftCZ)0I>XJu_|VJ~HNVQh6_bZKvHFKA(MZe??GE^v8mQcG(aF%Z5R{2zurHQ;X2"
        ">mCYQVv|5fLY%foGnQs;VLi-@ls<laN4s^L(&!)<N%Or%P17u|K6QX57xch%M8~jWv@Y(z#Y{Y)p%&IeV3U!{<+uLsYyW)vaNh>o"
        "Hcd01&1VCr07uMw@3!E=F>!`REnn}pZb%0GjzYlJq?lP5<@9v<g1%sODLzp>c&|we4ny+x)UzVYh3P)!Cqnyc$(*>-&1c4=2&}&^"
        "zfd&$8{|kL-FybREDb;t8-19L>Jsw90**}9nJPmz)l4^fSb)WhLV3gV;zVydkjx9<5;x9{#e{SLA3y)+r8~1|SgTS+!*wzT;))Q7"
        "J6QF%cbn&j&11h_FQD$H3NWhUI4~+pF=I1TtI}4hC*<o%;iUpXz~}}NO*a(DI#QF^R)=IVS<r_E8DY0Io`8JNRALG)VouziL3?dt"
        "hu-~Gsfjh7`X4Nq6pqMFQdF?5CvXBWWk8T(jAlI)e+pS+X#WY|TrfJJcgOf<Fk_eg&}IS6-_Z8cK!JXeDV1O}!z9IL)xn`E#_2-S"
        "s>*&iooxGt`(i%(15ir?1QY-O00;m803iV8>g^7s3jhFcB>(^{0001Ua$_%IX=ZO@VQgn*ZEayMWp-g~bzyXAZ*DJcVRUJBWnXZ4"
        "bZBpGE^v9Z8t-%4w)K1N@;{*LhmxzzI9t2hsZ~1HmTTQ&$!DqEa&C?XlaR$TMX~^BTlMn4?>zt{K?+ooZc{E22|T>_@P6aLp6AV}"
        "OdlAz+R06klUtU@n=R${M8pM8^ScpYPi2<IsU&fcFnD8eCDS4wk#+jS5|XgI*rqv^1sCDV7cXC&5K-Nhyoi|)uO3+bKvF^SLJ~>@"
        "tmUcP5kd3hws;E3iU}zQuX0l4*^WGJSWY<mt%6~e06Nd}5NEnA3l12BY@Q3zJaX1NwxpXO$+qP>&03hMN|To_*1Xt~gi0D`R0t+W"
        "LnBFJIcld2Xi4QJOK)qiD|kiD)ZA^lE@GP9vF(<Iagj?-W7%N5W@RB#kk4+J(^6~-=~{;L)LQl=>y<Ihi#!F8NSKw;hKkMLa^NxE"
        "SX(K}Y!|7)NEQtaRH@=Hs`7N5vLs5<^?C$KS+j@(v$#ZU?&XWWH)8OC=I<<@ta!zOmoL;9(Z;i);{KY6DwE@vF9>|nTu7S7EJ~9x"
        "5t8ds7?YK-1~g~LB``52w?&cZiFK97Xk)PP63FN&@^3OPa;70TqoT+aE;1o#B$`4}RMIiZIJZafU&7YP##vM<)7NtH-5VGgOh$rZ"
        "hfWPh$r;*DYM3*l66`q=F;7cr5C5rNq&d*OMQ<}^1A+gs7`&mi0kNTf{`Rjn49&9QF+vB88V5B}n~-yw305<bZW)Lu0#7XR1SGr8"
        "3Mv8UFUNgyhN%a}MZFuN0szM!SP+n-H-kp7?o%V*Psp*ERZ66(VB{B?Rcy+6!F{h@u&qE|x1bsnG9dSY2IqiJ`1M;rzL*Tl>=1GG"
        "0523p*F_y|Q?W(ksJW1QXCcyIa5Cg={$aQCTZ0LLS_3gFCGlI{Mr5%x4{d-8S6W!pr!oZ`0Yoms5XjynlNHbRG~`?MJtE%K$?EhX"
        "IzO4cot}9kV=wg&sNGj+gPZZU$H(tmO{kdeGQ5w|Zy|c#2HY%HC#%_Mw3xqr2h3o&8EJDPb*^eCkt7x9E=lOM7`qHwo)^Ma($Wt;"
        "TRyY-`SnROy}n*t?{QSrnG*>SL#Q^CFUe0umgqP)BAYaUWT^lJfdoimP#eo-3W6p%14kq92H$eS$QDGJfn>@`h8o#pDmP6s3s($d"
        "JmtGHP=1Vo2rUV0apIZ0aD?`J<SDzz>6Y2sE(-K{%waxG1tDjm$R3y<Saw}-f~0^6ZQ~)niE&qVm?{93uTZ|~<q%)7icf))grz40"
        "3R3}VGo=i0A%!#O(fg?DBBF?in3he{uz3Y;I7m6%-X|&d^^2HjowBDCjHS4*bKKtAM<`xcbh`r;>aD4K4cMS+0)6>jCD%uP^8#zF"
        "Kn3Rv<FNPR?0j*0@-~|O0$=B|`D}SHJ&Qc@wJJ$M{QAZZ!VP<})+>vG)keU+7a9FVX43_CqJSYUEO)jHz9j!(tW@T;hKemCSm=Sh"
        "RXL>7c!Qoq=o&4M=7doW1sRA$q5)`g#7%}aK8R!z?K#}iH1`pBqQXXiHQytLkO-ks6l6`)42sEDV6<5MRUk$IktNMCHD%%K##TMT"
        "Fylb&hq@EYw_MuVoQaytT`uX9G7DEgJmmEIhW5!}V3va}(u6cBA$77x)wChBOTv1s1sS0`3fS(mR}Xm~S}+uI10tC0`)iIRI;abU"
        "^+>aOABER<A@I0V)LZj$$w)1jsjAWCa9!@gs)VYLIpn3VLx>8gKu7`{saN%~HgCuu5)Jh}kedc(0$$SWdmsOP=s2*EbEuI66=xMS"
        "u;41|QE+UvmGJ+0n@GsL#dc5^**_b0FVV+ve{(>);xrdPYs+|FV_4ZffM8^F3hCJwEQ8`6S#hZrXTv*|GoHqgTBQUTIGjO5$TJ4}"
        "0g-fd5n>gu6>=gW^WIj{g%&@Jl5j~#B^K(#BC1!o=!NKj#ti0|**=A!S`&i%!tT1J?$8fv+aKPlBHF{RF}IfG67IT|uR}-pQP}R;"
        "?y=wrI{(eRvb6ig%$hksbstiGhjp-rA=!HmG3dsPRtW>xHmiMDH_?kmM2TgBv!{jHt@L$p9}wz$_pk}Kh$k+}U6euc!c0A2MKwpt"
        "c85XmgTkHELif{T(&k1u2FLf2N;I&OSUxGZw*^i7AQ&A+3o;Os8x-pMk3iNkpWGaRJVZ|gE%&rT%0`s9$PN>-i3N#ClvS=%B|by#"
        "h9?-X*t<sd!tVr?Ak?0(vW2bQ4dpo;HTHeptNSfG@oV2l*B*?9x}eJi<I+DKk!MVTs-XLVe+c&0co?)?+JmaLxZ`vijQ7`Qkznoy"
        "1`S-O2i(Bm8U`N96LpdW7t$l4D$B2{93#&$QMpvsbB|Oks~8Vm>nihySW)b#(sa2K5bLMlM*eHZUM*D`uJ>V}z$r?sd+xI07QD{i"
        "$FuhEAmqozwuD;Lf40m3WIQ5YeXma(M@>`jNNw`XS1^Z5zY0b~({Uj$c?ur?To<r=YO{-6eeBWYF*dM#V>nP#V`U7cH>qo5yL`M%"
        "^K=V!D9ayGUgTRK^&QBaT%5o9DU}!1?W>ckS?8F9$u-|FB$SY=cdLuV98RwnRspF_q{<PV@nC!5;gRy)ay7lYUMyDl_Rrb;Y=$MA"
        "ce=P-UQJIueFnam{!7ijJ&WGXe!M=peitbOR{e$DOh62NKi1(<2Vu4|(GsI^AwsjC5Hmh#PK9K)gMy-D9AERA%Pwkn#c{E?oK70="
        "=>?0Qm(%O3>%}=7-3MV-msi7c(?gS&S0JdtX)XTw0(n@@R?~_1+H-<joGdSv)9Km7J9Y!i7Ss9Z;%qkmsWy|5)6PJSL+z<9+8PSa"
        "O)L~}mWek$_kK;j{xyUz|A(>q4Sw)OLm2mK%?7Cn9<Gz=Cm$^o&d)wx)Et<egJCuNc&WU9tiq33k7tRtt;DOL)O?_Z*y~^($a&t;"
        "tZxS0>G<HZ$vkC7B?i0ibeyVR!>$$~+YccdO4;Cu6Zk(c8`Bb0%%Y-_WhHgvzzvCU@Zh{|-%Va0I{<>dJ?oa>A8Z5GpZ~VqHP8#J"
        "KQxvV#ZE_Um%1Nc_ucj!ss__+sp}<^cO3>;BX66#;T<n3z>ACYEKs6c3hb2mdr;U9>34*j-K9V5!)}PKJl!w{%*1rsT~lk%!~t+w"
        "v?fwFs>f}lc<##q3Iij`XZ`yjba7<~YCm=~Mb}5e#?5|U(EhJ%$ux0Xo)|wS6HGdesbL5GH0M0znEi6a1%eJBCLU)Hb7JNKsa?Bw"
        "|IvfoZ(a?0^(S_)x_-QxT}~H2uNogV4gI={t{T>c!;WJZW=yR4+%DSxAeP{rpUK^-{I=qQD$enA-Skz{yav<J#;QuyeIqtruTGXr"
        "4<nAj%s>#t!wU~9bHcQDshULdLY@~@o~ZqMT?)W5#v4!fm<AwvqG}nj#nGXxy1QSvTk?Iu@86o$cL9;r2S<!>kfQk=OKk0a{*{bt"
        "GyO_&-<xo#FLGa<xnQuCNKYghN;esL=)At4b(9_&XY^~c0Mq+^4W|pSVaXdWxH<af`1p9-4>TR6IbZ_os<_(12`c@aMY+D8&})Tn"
        "e#azBg}mvD^}V_w#S6anRy1R||E!vUmXtevLxb~!y30gxGY}lSzf3CpLk&1U*k)jn5VaXli58>FV;j4kZM+Y*Y^K8Ag*;ZkO26yo"
        "{A{sp@Tkl6kK@5~ugu~QE@QOM?e@K96W+Mpkfl4%qNP~@K>c7t>D|JVei%T2lMx5eHh|=nB}Pzc`=X+g+55yASGL{D<{sn)!T5tV"
        "sXs`?iFjRQSrZSsLLzVK7`o48S2XpXXY6XOlq$o{-)(5U*($S{?gTl<sFQac9hd-u&#vH8^U+2T^R&lGucv|cG8z(K1jKKa_|FZV"
        "{^}3b@((tDW**qP(SI#u%sX(~a2-+Lu>T)WO9KQH000080000X0JanEsMi4i02TxQ04e|g0CRF<FJftCZ)0I>XJu_|VJ~HNVQh6_"
        "bZKvHFK}{YWNBk`X>V?GE^v8$Qq6ALFbuvM@H-GX*;rx>I}IMP(^6o=&=u>j9lFAhB|2p*O9mw`1-icbEIVzSEG<tW`TbB5g%Inv"
        "?@Dk@LyDw<%U|ndjaJ>jm&<SGUtvp(&^T^o_y89FAOl%tdX2DgMuX%)ja0!GLWnd;6KXrh0dBpsY0|K3K{;bk1#;FGvR1G|d=AL6"
        "Hn>A|GTa!dpSl-A8s)}bO^s91Y_M%*p_~mI9iO1zv6I9XjG9K5NJ|rq4Q!bLR(^})jjHY$;<MSgd>-ZMSn7H)&Q*dB41dR=a#((Z"
        "b7!$k6BvTs*24;}uBX%^+9CBAvV5uxj|Y3V;q_qdJz-){GiwxNeUMgRMLJTV><l{^OY#UmrRj0a>^N)0r@k|!WI&kFOL#b~`zZ3g"
        "Vl*hE*4b(73>hz6$gV@Hv%`tvpBK7sJAc%O@$i2shM1X%@`V>zyufXpMpIeYV&%AsKXoXv@4v1vFx3%tES+FIo~WVqJ!2)kB3hkE"
        "<B#oTueUn}$r}vAqQsTT;^YU7!Jg5EqP^3^*_8Oz5AJaRcuzi5?ruyES<}7;ybqc9ZAEeG$Y$}sa(Dw{DQ?-gu{cZ#Lga&<RZf%L"
        "tVAE0<@>l4{{T=+0|XQR000O8001EX6y<DOKmY&$LjV8(CIA2cb8=%ZVrgb?V_|G(Wo>O?FKKROWpZV1V`VR2UukY>bYEXCaCuWw"
        "Qt}MZO)O6=N>wN*%1<dxF3HT#Q^?FqOD#&xOHNfNO3h8o%qvz%1PPX;rWYlG6(r^4Cui#^DJk)C0RT`-0|XQR000O8001EXDv?LZ"
        ";0FKz<Qo70C;$Keb8=%ZVrgb?V_|G(Wo>O?FKKROWpZV1V`VRAWo~71VRUJ4ZZ2?n?OENA+cpw^Uts?SR!>r@2zJpHSLYPmOIj2M"
        "1c$WOmu(=>5@i!Bkv@{z?DpFKy)*ofC|Pzk#a)2|&UsjyAM-uGnPHM7`F^t{`Y4ql2gO+PL=<b8Vxnf`zywp&a4B99A&FuANh;cr"
        "zyCyPS|1sainR2e5nW5gbde;<?al2?t7K0aYG~b2t(hhq@0n={5BW7CRec;U?RhI}+8tQm)1sE5<%e+j&EttF-ZNo-II;S|d824;"
        "Mt<+uiE{Pkk@lyK<^JbC6l-{mI<3(_5ep;lnc%-z#RIKYs-Bt(%px16D&)l;O-?#Vo9Y21YdLEwf0L;-+PbD~E4wC;{FYHODE3yG"
        "cZ@bqZ<LY>EpdDEx6x85kokp)@<%nW?Dod4k?*XZz8jshyS*XsvCJ#p=p8WwXne;72v`)wd$+HRJXMNO4L9J2a;rfjdmFC8CAZNq"
        "w{e1a;E(ft{MWG7M@;38rtV%3U%Pxxd`&2S#FM{r;)Va))5l7%53wYHjnu0INJ-LdPPUt`bMn<Di|80>XjcUlBZ_e4%gr}m!_kbq"
        "e0|B(5a@5LYD8k5P|-lv)UrPfhCyl!8j{2lClIz^ZFYJ^^uP_>cMOAxK*qooBN#aJ#5-wx`XG7U5^v0Ny~$t6cG?3%LYl~kv#pMa"
        "blS?u=Sy0zFB-$q$#xSxMY(3=C)y3nhH9Ej@KEiC=D;959&4}{Ym&^}dw}TQ4;@ucU{@O$g=gclyRauJ4LLKa-gA+rFQU-N$%{yf"
        "3@org?LhSF=U$hQ09Q2`^A>MbjjX_^Rm)9fWv(VFK`|85s^i%cuE|>|Tqwj*e~s&&YSwYVs+tO92=T!91FPmK8CW*#fkUj7L;&!l"
        "j{Au!+B3tHPP43VCq@C?&%!*N;8sZzI7pM{I#jWL4ID7N0@MR^qa9eMsXj1%IGRx;u=WF&gSKWGJq#es1s8`14hr@_yVT#3gMmDO"
        "<98mI945g+0s(t4$ExE!H$E54%yl2QIl7uraN*08<(|f&>Y)cHx}yfn2>RVcj(MMx1}#&%6P*-?+*_=4f3h=rU<-Xe5(RH-ko2CV"
        "ur;T)m$F-2*9PpUM;kdwLK$%s0*7ohx-iXcgs~31hTrB4E=A<REDKf6>1c4gE&e6xxHd6<%nu@nRREa~N8-TJIkGIhVI^$znMVS*"
        "Pj-h5t#x-;B>0Fs@Z*pO2mo?XW0cFOoFs0e!aS=wdEgN6`3$}M>IzM_0W^I{w6dU_xsQOz&3C(;yxhKDF;r{4jbf)h)^!UhWmxJN"
        "7qo+PS~D_!&J$Z8*dl={N^!5+iWb&~paj%I&#|EDfOz|In-xH&2j=J3Tw)jWbYh}OEs>Q|pq--+p%V)C1wBSwFkp&mF9*>e22Uga"
        "*1p=fj7Wf1vQB3YgznF`LhP*x4Aue@$H)_5TROwNEPxm^bK<OFNNr1!I@4si*D1H9IfA1sUl!|Z>avt=T%i$z2E*W{Zw@px!VpiP"
        "znzMhZO)+@=EMVx?f1y94nSPi=eG3EOHGNb4|G0Lm-CiEZ;L`h?iz>zsPQLi>f<Z$9)fsJpP=7SP~R!o6U&id9sC#?h!SjyjN*v>"
        "L6E7(@qIuJ7*by9qEpnc7*8$~91SmE7nqe7n)K9L$l|JB8|$@=9lv<ivts`^6?`j<-;DypuRKGM@#MN9=geK~_%(k$4f^JUHcf?u"
        "p*jZ#+BqSj2VyTH%{P6EzJHz`=Liyxf4t|zj|){a-<)pp&Si-d7$js6)=s41ZA;prYk3F#(192jExCY@9qgb+1|3#IaSP$}vq;X$"
        "VC3RZG#91Z;>8tF%#p&x#~-0!n<J3nS`X~GQed0yBd)L)eOLk0-G9qMAi<wt3FZv3WQGcdp!7W(G7U9QA-;A`RSl#nd_w2|BO<Nu"
        "{+%e<v1CV*K{ExVHnDOG153_KzlBlI68FhdeoI06XamzB3Qu>{W<(w*7*1dnxK-SPA!kIUt`;;N(=TY@)6b3lE2(EGWfNRXa~TdR"
        "?%UnwJ^5@h%pS^LnA)E`X~CV7?07xSBEIs>1|i2T51gWj-BDr=#-l>ATfDG@(b`4wZt<A%v59<~*|BPtw%$aK6chBtTUTn;#BSxe"
        "Vj9`Qp=;CqU$)<T5vKd)i_mU9+>J#7{I`&MrMq@sTYJ=5rug@pEk^kd1Lg%}92N~oH~|iRV4gM3k)e)E%-yqF5hveb9z|DTh*_6D"
        "Go;**i#MvC>mCCm#FiaQ7QqBYyws6|V2TrN7BMR!5bgK~D8uX8{0(H$tDG;*(>O#rHU?|&RAYF~&><Gx>wW1Zrkwv=oLh^*^3uj!"
        "a##FE`HS&kZ}c^uzzBiAcQ8U=D;+;{MUHp0@Z4^kHEilf_#DqF`ktTGk!=Px%gjl~iU&oB;!w_^L9tzrpglC3mS?ttLk*_~A+_Ok"
        "Wt+9AwLO1Nj7r`dG}h7O>H}iA-F)%oMPw<T0V!p`lHGr}rFCx?e}3zEspTAByFaPp{DYll6cBdCs3a)WcJjk&gBiDppOfTg@lPO!"
        "t%^o;MetlUYsvL~?2I7vI`CySBK*I<`CkXuNX!+Sjj+;(^vWR%z-yOR068)OcH9@CeGamphR=CKHu|rg+ne73P)h>@6aWAK2mk;8"
        "Apmvagc~UV0086y001KZ0047xV=rQ9W^ZF*Y-eR{ZDB8MZ)9a`b1z?CX>MtBUtcb8d3}*hZo?oDh4+?x2gEwc4Z29O$f{D6c2`vh"
        "!^EZn1`Xq;zJ0M%JN;4225H{oW8RFEa_`P)v_C=!T4*wzV=&y4_9CF|J>-orJrP9qgJ8%ER$Lr+A`g)}U}s7-W~988Qr1;noe6_z"
        "uH1IyZVL%~-ZtcH3&Mqj#Nwz6>Oh$DjthYu#{FbiA((QhmUQY=B785MBF((r^UHBmo@QZ%1EB$HU2OCGTBqa(bwRRf5miF(%&@`<"
        "KrL9M=KwS91ZOBCGw?VWmVhg$sM&oi!)bcwt-96}yEv%Hw(_2^FTVEHLBaz@hEF~f1!yCt+m6Csv+2fFDW$zvN<4_yx)SBcJ0Hu<"
        "w|;kPi{oDQ#jrB!#q)2OKk$Eo-IU+?#`;@beF0EQ0|XQR000O8001EXIFgjua|ZwbJ{JH0ApigXb8=%ZVrgb?V_|G(Wo>O?FKusR"
        "Wo&aVVsdF@XJsyMd97GmZ`(E$eh-lUKx8jcD=U|^d+^`^y=5)BbzPca1BybRCCcWkTSUo8`mo==?~syA(Q?*qGlJG455IGGF5fxw"
        "$;rttRU^xSXH1ET*SrzQj5m!aXqD9>6}2d0!P2s3e3@}tRsk!@hBa%!5|Q#YYZ$NNwQNM(w6zG2kB*P7X;ao%nSHptW(}`bqG7EP"
        "isb^|6SmWm)@9CQAsfjv`B|%Ao)Q2kV|DuM>c#6f*RNh&|915%%oAV3zv-ocw_*!`@nY?rg=bIwfT<We#Y!6(*QHXU8!4HzxbE_v"
        "k$7lKl||JyyG^NBTf}Q#z|il3gXg#uX)0q09NH!aa8`*r#sXgnR;H}Nc~Ngga(tACxJ*RNmSWAf628M9{HWJfHBSgeh3rZ|*^eU6"
        "IV@-dcC?WlzRask<d75z%se?cp_^l|tQ(f|W_^65J=fe<blYyDkEbQul!(swDJ%-cp{3Yc4pZE`O9@C^63+2aoN=Yt)z2T^m38hx"
        ">gO_PGvUvVj~G6%4SO)U&UYb1uoH5#cHgp=$x3|R`E4s|nM!P%yJ!3!eyK&3@mOCt>@G{#NHG>glpaxptjN*;iIvaEVF44BnzJ-3"
        "d4nW47qjnri|C_T#K-f);k0k<0Vu?29K*lP`mK<wH6h}n?@<)FXaHn9R2D*cFu?b115qBgyiPj%?&h64uSKEC+Gn#LMhljWTV6{}"
        "1-4*!VO8FG7k-!vUU;+ffNg}RplWelx5Bc$7AUiV-LXd<YOQYS#v7m>54ZCnV%^A&<&RYAFQ_`6^~LdA$7nLTTAFF`*&xr%0hC5j"
        "pwKA^pG#BpCVyUa8P5xx<*mZfOH&wO@4RRAN?Gpc>L$_=el3$k6clZEjr5_$bWIzmzQ#NeuEL;8r6=swljp<{Jz`gN4<lA}S(TKo"
        "eT}-97bDD~&7-y833BgUn=c^`oE^vLElZr9jsBN|Vq~UX580cPy+cw3J(ns3)%F8BJ>4m^P!qDbZRTtjk8X*sv*b&e$>!cV&@pVZ"
        "LM)>#&*rRSQAXs+XqlxvMoNVUli_t#F0mx@EP`c!>9w=7Cw{O)*TrKIFNdd-9ivhT@GL8DX@6iqD&i8HP<$43*>MC}|EEbI`?~bG"
        "CMHQrVd!zyqBKE{(8W@D2W<!Fc*Mdx_T6EFsiK=rMD#Uhzpn-CuPF+=s<Qi@^%xxdtG>ru0HWZjKQ%L0o3wEg4y7{5TVsL=4kSEL"
        "+JVd+Z&?Y54Whp9f~kD#o54p&eS(-yw22hz85D619!eNX?rCtqf6KDWAVQqgiD>>K)nCKSb_FbKH@O^M*mBmBBR9B=XubdL$n=hb"
        "$naIB)_0S5bQLBtB@ULUgDb(0y@if3kH(svgH%w|8qUKrcB*@RsHuK*#`T;P4IH(e`pl%yH|+hc%p8E5h)U43zn^3wc<osi4V*g#"
        "Y$>@~Tyi+XHD$ApWJTy}khcFD5)ytN%RN52?+D9w;jeFJ@3dcsib?lEFn1`@IJgT=KTsWtr{%~mJV9yspi12NO+x*mx%VE_<On)m"
        "M7JxVzTar!m*&Ci79Cg70Kuw$mYVOzebBRN<28B)m_;`kq8lHWF8$bWrXAh+nv0&Iin7A<&N2C5>6GEy4Bz}Ql|`#TT31q5u5bpG"
        "^>|N}0m0U6f6L7$-8Uyk#YPW%-bbVS6S4}Ux>%S^O{L#nOplTO*e+B9t!g}GPPC3M6wA5VSk1zgcDlwxqbz#grLiA@#)L_)jPC1~"
        "Y4M?*8_oncbxWg)db(QkN-%H9n|Q6OG|EIVDp%i`8C+K!!h)Q^vS{TH*4SUV<6&f{=>k&^(q$_ciR>#m*MFTVr~2pv1jinq-C`IK"
        "x}Bb$BO4PL(?#&&kjVt0*a$r)?Xa%&+kxv+Gr+f1qW|wl=iVt_=T<%B^bbo*KC(KsfetnDPCBw29a;T)EWMMz={u(BN!}RU^5CWy"
        "G%d#)16zXIUT}Lr(59IyVLXFbuy@YTx~b>fn43)SFd2RmZl$>OY!8jQe=u(Te%N53RrDs}Ax$ov>um<O!G-@8j-0Ul?{K!Op6OZn"
        "Uvxw(#I)<(ICWt~R_||I#RIvh7(=-=%;+9v7J*?{3(TLlnRkPiYwJGrvONPA_v4tJyKFVMhF$#jB<NhnkqqHE4{d6}F6R67{tf#u"
        "MM9cozmzjZb}dGaALl{$ZfX9VfBFYosXvje##h8^6Go5SYCVMnB9bVTS?2v^BM0Wq!8@7BB0ce)TTIL>59&d~w!!T<X({t#2OYW$"
        "Rhc$9zw@l)`uC&7?A$+iY;VHxYyl_#e%GsdjKSN_1WgRZKS2{3*Njl3{IvB2Y<qx~mOS3r+%#w!CfR9K{a3f5!?;8xt=h6xcr}|Q"
        "uPNy6jUAyk_@g=7>$Ocxgw_uP{f|gxF=wBF@du)8jf0O^|65hz{SmO&mUcav>7Qo@FY1HCbOz=soz7uWeg8MLu@p?Oz#on8d3^LY"
        "P)h>@6aWAK2mk;8Api+!qYA$P002V+001BW0047xV=rQ9W^ZF*Y-eR{ZDB8MZ)9a`b1!3IV`ybAaCwbWOK;mS488}*e;|BnfyUWs"
        "2OowFLkkQ;cgTWW4Mn13A+jitOgit^kB*zAD>`7h2o@jTgLG9@kMG_$P?*BeYorX``5xsCLKKbJD|Ac)XpcZCKuqZ<%1+}6K`?cz"
        "uRj1Sb8t#rS5>uKEEngJ;MDo#L{K0Og;=joQ_}OBmHGZp(Jpr6UE1R?xLUmHx-;sW`hZT4jNeE%=)!o<OgvRC7aw1AK3{O3Jq6@P"
        "9Eitexd1z!vf@u|yO{ULg=5wROwAm{xuP}s{!qw*E#oPQnerkl;QwUxF$B!5s8f_?)>v;b%Z2bva)X5=T6Fw8x!znLI@qJO#O=)K"
        "W-|KjpY(5G_2Esu)3J#>94FTx<__5!+8}cp8#v%-El<cc9+4xiCA%U$>00T4IsQUkKVjVWD(g?*w@@fWKh_ibOY;q`W`^hez^u`5"
        "o@s*X6d5o?l^V9}Vm99MdV%vyrk{!}&IELADT+nF<zOsv)l)IvCzqlPc-l^90GZUV+%92&-DuEJ&Y<kP)_*C#ei*#qn+tR^{Q=b-"
        "R9}XYa4-?C5RJ;@|9Ylu>rPMEOzUH*tJ~$`H&9Ch1QY-O00;m803iU#MspTb1ONb~3jhEv0001Ua$_%IX=ZO@VQgn*ZEayMZEs{{"
        "Y;!MTXk}w-aBpdDbYFCGVQzC~WpXZXd5u=zkK;5Dey`O3VK6TV*A4K%OGJ5r10exg>9qIKT}5s(X-yM5vfb`2OaFIf9H&j%9(#U?"
        "G9J&*Z$4WH@xfbFdhqzk-x=M!fRi<P)XrE3-b(Fiv``xhev8mqB$?7!fwbjTc`W_jvfQjOo6qL6kFrsf^yok~4eap6ee1TeqyIIM"
        "6$GhYpfue}27ejQ=%0<<K}pIElqM(2L2Igg!{&q#Y*V$Ju^ueWXSFpgbkc7dwT39ar{9s9Hxyaa948aESyY$~H6NMJDy&aw7F>m*"
        "++w-wjM6@t&w$oimb<mlxCoB(;BNunEWqtlS5W+w%VBLy155Z-HjWi7){WY1eZe)I?#^c^-2DU<l|Anru(tX{`+_pZCTpbL?BxcF"
        "{#W!`UfK<M(i-B)oTow;Y$!>49sJZ$N2ohXc<m*;s0%q9WV6S!j5|v5MK2Gf5B4+aY3l}iyi^U^cu3CKQV%OW3`+_b+cj2&H#;hG"
        "_Kcjgh^LSJ=UMsfx$KGaHsM013#hGZkpZw2ofiwJ@SsXuin7~_6r`h;%L;9N4n`iNYUH{>%Fv-tdWDQ}mZYPgTjiY68`4_E>b?sR"
        "$Fw=$Q>`e$i4JE=`zKdVlUhW}qnLQnP&kB-T+8>?8k^MO&ltiPPY|J)!)vs%%71t|iuC{5GK4ua=E89pE=n^Nf;w2ZhKiVsQ%ZX|"
        "tByMxGxGbc!6fXa@YOQ@hXAr(dg1eC!q+MxAMqgsNcpzXm)nrwD}{8-!I`}+5nh9j<6;2#>$U%*xVC<c(CN?xkP_KML&_B`dKvs*"
        "*EUEPL2tEH{H~Q(#Bu*1q1Ab%rB_CSBi5mb9Y}86^GGh)xXo&#Nv-0R*f6cD5y*JZx_*XFO}sf~N6vgSmPqUprs4uneW8np`xCD7"
        "D<p6g&&td%);k{Gh^7>-ti@n&Afz;!$}{F1UV%17HdYN$!eBjrx(exHSijg$k2ao*tll?G_`QKqD;oKN);(s)WA!e5%p5j#A-&fm"
        "*%frJ{r&Hu|Lh9tSxbNXA<fu?1&25_d@*9Z2a1KzO_ptPzEGlfGBRPWnEQ=*Lmt;+#yVeEVSDxcQwl!<7q&N_-agQ8t|sT6&IRC+"
        "*&QN8i=cdw=8a}dft^XhhOQE(ps6lH^-EZ#77x0WVI25!jtrXj6e8Z@fC~@dA_~ugOq*h3WpzcTyiOcE9`LuBcvz9vCY0QVJ>}%v"
        "r@w|@lqEOt^ZmnJOR0yq!YB}aJJAdq#rH5xJHek@Nz@6y-9NznAMaz4vfB<uwlcO=Bz?d!9(v&ItscJ~M0geTp^H3Tm5a%KxYY`>"
        "rm!NF?}jUN#KwV}7?ML3(_1DE@%6}7KC3*n978$_#;T}GdP3QrAB)IUbmB~!W9=z;8T>IWK0$9stdgn9BrHul#FZLyiK={d-zE&s"
        "&v5GKO>*Mq!7@Fe{sT};0|XQR000O8001EX`6tF3a037Ug9!itF8}}lb8=%ZVrgb?V_|G(Wo>O?FKusRWo&aVV{c?-UuSY*aA;p}"
        "a&Ky7V{~tFE^v93Rc&w6FcAJeY5(C=B$S9WbTB4VR3sQMkYFI#H-sWLxweMH4z>g9zvukY#0dpNz9hEqo_l<E&t*27J^OS6ci~c2"
        "2%n7VO4BKX4@!Q;(r_u@ovgYVVIvj1m0~-;l)~Vx@gA45X%+5JXfn-DPfkyQrv}}%ZOyR)8$eiNvu_dJywJcz1$r+GyM+o{vjZ-S"
        "q+P&fja9|P7Ie!>B)}VmIM>`5R6tP`bEGoshoVAJ?g6VU>db*&Vbd^+hFirPdq$TL(g1JT8XKgOicn{>ndQ1svazgeKr)cZ!28y^"
        "vDzMoEhAOA8?SOf$Qp#OW~V1*&9t`nyi%;)g$ziQ2$B2jrOVS3poilABR~I@91r?TZFRu6kLa4h!W9l~3D+I@DUchexHeprW*RZ^"
        "d5ALoLzZma)lBWvd_Zp4ZQ&>_;T^l>P1ks$RFcxU2f{?xagAKvx~O=wj9ZJ=zH^>{Mw?nVd-U+awc5)E&w*YK9$hT_`uurZ*9f{&"
        "vXz}#!VCWat&(fD<~29_yifZfSru-eD7fIJD3Sq0W4+1#oI{EXD@!Ijnhk$^ah@H$aByCnKhBP@-qe!W^F_W0$0^L8!nG6_6_uI="
        "Rg&gC^>oZVA2TMr@y5iu=T3(G_r#{Z0onB5hcw#f4qP%Sh*Z2}^+e@;uTE(m7!}UK@)Y^YgK;t@oego*ap2b+6(b$G$o?XOIuc#V"
        "7VitR-M)nye_%aLm-9K!q0h&k!@uPMnX-*`=;Ts1YpRUZrkAc$vE>09WOk%ZOJJRSXpffk!J@bstU1%GSFF}Jy+Y0Qs3=>i2{3kj"
        "P0JBdt5;_@zxeI`K`YKP2qy`fX{;5}-cP2gzepvLiJy%(q-m+*Uboq#<HKzI9NT>%4z}6j=8w-K5s-}S|E4q4;5z)OLg)3>*KS%U"
        "Y$c5{srFPE+mrGef#q=(CsT8J6(8JA&GD~`HuR3sB3Uz2Qjci-UeqY!IeF~PU<Sa@n7(iS1v<cfM7pn-k`CjP`Z!9-Aci6A(Fk#m"
        "=c7yyi$eobp^j-d8PYbA3EzYy0N3spJQ->Bp&w_{+VI|R6)Nwx6Tg|kBAX7aIh>_qponAtdG9x=C*yb{x^17w)gaL10G9s6a1dhB"
        "pGEe{m2D)uk=-v)O9KQH000080000X0P5!>`^W+S03`<i03`qb0CRF<FJftCZ)0I>XJu_|VJ~fOWMyn~FJ*0FWn^S&ZfA2YaCwzf"
        "QE$^Q5Pq-J|8V4m7PZ;LW6=<3R03&3sOxw^sB-P=VsPwmc2=@q&pAocrE53QmsGy<ci;Ej`I02LK7Ts{NzVJJba~+{+5lTe3PPpr"
        "-8=YpH3uI=<-ubiNCZsb1`(8#{_Nx8^7EJF{Bn7_n5Vf)lO!1qhJy{c9Fzz`8sR;9(2PuBe-xA2v*1W(skI<HSQ`Vw!G|t!#83}t"
        "FP2o|csQufV4km0DQ5T087->?`qG5iZ~*)Z8r2G|bP8#uu>Q_q?VM@iUfe%Qh=OtqNPY&|hBhS=#5vxTByq;#sbQsX1EMHQwTk|%"
        "aK~a*)HFuT?G}2*V$zQ2q=|KWPRCXzUu7s%opYrv3ThMHOge?f1#)~0)A!HX!wp($gibfO^)Hmc!x}e^5TvuLx0HdeL&y+Z5p=E}"
        "kkV*Dg!t6d1nuhtVP)nhg=NME%5k%;3z1|!IgVoQ4d!vIWQOv0;n-*pMhoBI5qh2h5wx=@Tnb|tJ1>2JjWfn=Vg9#3(h=aMK>3>I"
        "u*#V`X*qS)RM5DecKVt>dUNP#x<!4Lg_PS{AiTQR#e3(iO*Jo}J;HZkN}Llp8YMgBSJw~$8<rUl6#Te%QmjiOsG96>VXIZhenO3#"
        "CgYR5)@(&|Gy3+GAK<eb7Pvv;_G6#L8X>|=H-x=f$A)DO&k1etf6X~zkR~3cN9%)|`={hX5BwDd_BQd{D9E%@Xd9zC)-AD7zt{m!"
        "h4)$>BiaP|cX<d8uZQZW%=&zPdHkZ{n|fD3w+npJj*`>j<M8~-pye6`*Z4i9^*(}&LxQ%<(Tv7VJG2{=K5;<970>su2bT_owyTOS"
        "&7M*@e`Y{p`99Q(7TwyoliO#?10hkzI)XR8As*nWUAVhSXS?^)Eve41<1hMzI>JzrJ-!V`N8ZJxF<=78u8a<`@IO#X0|XQR000O8"
        "001EXv;=<25(5AL0tf&ADF6Tfb8=%ZVrgb?V_|G(Wo>O?FKusRWo&aVX>MtBX<=+>dSP^FZ*DGdd8JiLZ`&{oz6Z#E;CxDf)ahZn"
        "WauzpNQa_pk#$9{h9c34Hd!(#rCo;o_)&f*OS%H{$snnZua9JsB)1oD=O8yIf0Rz!042F6w8BaRVHM=tuq6a5f__4HM!nty2qdh}"
        "k<uhd#-s5lCs%+HK}aLKM-RFv9fe*LjT%{SBsZzGAUs&xy<X|uNnzGl6e5+*=6c<tEPj?qx<DJgl_-zddnxkV8P&eHLJ_0BbKxF^"
        "+AWA2MNi|=$1bD^!~8;f`9NDd8;>d(Tw5KqF!~p#>+I49J0FjLKkQ<dgAddQt#k@0TRFAMyxWq*N(;Z?&RN_ydUmAzKdnlGCv6ng"
        "HUoXF@Ebv>Di0YqZsX_$2jsq^jM)tUL<pQ2%d#M7jctzm1^cT;s6IKax8<k4DHHSIYL765Q(MD?D9zT#A{s6AkYiKeUi)o<3RXMF"
        "g=Bi{;P&EOORYidbLLBWTLB)c%RqfP<>Lr_Jo6(Is0*PmXHQ;|USX#9j3%CWIX~d(8JvH3flqeWOSpsxkv7tltA-}i%;$Vh0cbp)"
        "WJHV7NIR6K$9`877)Z+lZ=iV)U8rEIk>3l&bD!bO0df@9sD#{YdnC_Et*_`ApITAiw3qGYKTsT`)D|FU%-T1tNi|V*NlwDO($9Yf"
        "5Q{eeNw}byC?nm)%3#uK4m--l@B3Lr9N{L3#pleRAj^Z@6F7<Q2c4{p6RqO(hFx+tJ*av=*pUp4_N*s#$75tC3@LQ<C%dtJ1+!n;"
        "0`q?4P2AQqSB7(K+cRREP};7i?XvhfyO}*!6l7pN4+U3{^=Z2YmV_RO4L`@N&)oA`Z5h=%DrWa?xtJ}#EM_;4=W!wO!?;^GCmDPS"
        "56#JH`wQ0(-Rw8T<+EjjgRnu!wG{>m<Ymag@WIl7ZQ@>2n)uJ`-t){rK*5)jqMs)u&lqV9de2TMreS~^X?BA`D<}o=Fa&;3k6X39"
        "A(CmZh|;c<&7cX*tv+_xCh>$PQ%LUWDh_UWd#7O|>KR`lTHsK1u2H24GvN<VO9KQH000080000X0KZvPS=AB%04quW04o3h0CRF<"
        "FJftCZ)0I>XJu_|VJ~fOWMyn~FKuOQZ*qBGWoBk+V`*h>bS`jt?Ogwl8@I9l-GKiGR>3P)7v0+D7DaKX9$u2#LD9syI!=JX5Egq{"
        "O4{CPcfDV9rx<>J`<pK$m*jGFieDbu2Xq0Sq$N2V4u><J8IqpOW?xl(naS*OFZWe*6!IZmcOvaNS$26<UW@m=gx6hO$aW!K-Tdx_"
        "Xsb=PPahWIA5~s<qO6*ITI4^cSU-OH<mr>w6|_pBJ-=Isbyarqp%Y!xmuvMdhhA$5e}KfgsoGZ46;&fL*`@29%!Dl0RVEuC`!e0i"
        "Ml^Dt=4C6oorLcj*+2yui#Kvz?d!gi&_$X_EN`lQyMtF^1N2}_`GGnXJWe1uo6V5fd|y`$uu#=U>t|Or>)q2Qo2J?$$+871d)c({"
        "9@r_`cq{k&^!NNLFSiN*7OM3(^y@7Q`AaRxg-GkVI3`WirOh#^t2UAQTkP@AFJFE6<F{{<?_PfQ{hPlhH{XBzm#_aWmg2+HCjy??"
        "0lEbOvuiQKS;&$`4>mbm%@(SbnX%+HUEhIRhx!sFmu%|ay)d<DQj_(rFYl5z|5-xis~3ONr3K8!WZ@PBoI_VoL9Xv=Q~?kG)Op+V"
        "Qdg=AnN*uiQKiVotF&lM(cK|2B+~V|-}ePg4DyQ&I%e}DistE)Om0}sl0)9+w|S9w#{`EJDa}X@pnKT@G{*Vaf|NV_sOp8d2VJY~"
        "=hw^uh(5;-TaYnLbDY8iZ)N0=h}&IS%Xe2RBz?%&a_JWH%Q=%oapn>U8shJWgKK8sbydn0SEXST(y5BY3vs;?e_Co%Tshw;-bLQR"
        "I_rp#8iq~mw?S(EY52NSWVmH3i%l}{p}tYOmH4eI$#>VxJray>tE!09kZH+URGjGuw2I|Jot9a2dBw8=JgGUlyiz0^+4W5+qG5bI"
        "Jzz1S$Yy%s_C6;Xy3f_nd~z(Kxp9UzA5?ExRityYY!>29!XF?B`Y!fqdj|y9DVhP$m8Pnz9!7-(Wm+a!t~KvA?bf@6*vT{lTUy9+"
        "+wD+0vu-DN=8vp)Tpd;lRfRI>(I})*4KDl-Q_oV2Ce2$ZenJ*sHciz;v%irU=O+8U?ZmAVS+3{r_9*`P6PLy_C5A0n&>B1EM$ev^"
        "j=20Yans0FHU}w6Si$gth0my&DAFT{Ar@b;F#<7lu+YVQ)!empx|XQWWLMD+#A>|qsD|2AEb2Jxj<sH7ByiMM{0vv2zQN+L&(b0V"
        "E>U!m62BhBLXFUy2}P=1SB)G@1oIy`9grfRQfbYLX+*FRC8_!4i@8sXzxW(^3NcLfi`86QXyr2Vjw&@_5TMjnJgX$c23CQ=Z1SRr"
        "{+kt<EO0ct%*)M8b3ba%BDvpz_l>YwoR&u#@tu@)mhYG73g!#dV4>>&EMd_mMSdqEF0ko?PnD;+6`#|<l?`vI+myUPZVyNy2;znG"
        "V#!@AzcLSNO+cgqOx2Vm{l1;wA(=j(hnZ&H$}-*nDncWZ7O}D)kyCde>ZLI(pksLn>Y(3%nSbu7|7i}-UJi~Pat~w=0WD6g3y*DD"
        "s`d(TExo4p$a>%f&zxLdt(19Q7b#%I<SjxRgR&y!(EF}JkP*L9lGPR%FyE`ZX}im+0k=S?7BD6uOJnN4V3Z5?(1s^qF98Bdfy%K?"
        "+m3)Pl6+M)_i2-n?zh?!l0H5gq;X_*Y}p`EeS_i*@MU_0w7(E}C$dViMz;N4eD&(;526OQm8S*rrZH422&1+V%2L3;@V?$bD*(Z<"
        "8MEf~zrgWz`FbzAU6t`G^l}>>g5`ME-NRz|#nx{4jsDL9A>;^$--FL2^|#izgV=`mE_Ab)FbLFosXguRhM|pmF~4>O5x40<CNMri"
        "#Wcu~51+d&Jof3Ghc)q)!!#_1$TeP*LO3OeQ^N4Lv}D8M)Bps3D9pIDB{)D78w%o6=W<PE;++MZFlZu_1aX>Rc>H`}Llm$H3yc2T"
        "H{J;GmSUfm5tI4cVa6ro`57|CBZ=bkYefQM%HRVWGPW}q*G6yUP4T-pj$L4Ry$W{YHanxAagoPAPhbr3C0Nsd)I1P7u7D5Ch%(i~"
        "C{~MbZPx<Wk8cg^C$JRsGhWvA^E`e&Q8GDcf%@}FDJp+7gc~eU&?NvfFXoS!Ph(opYKjZA%n3MVu<fnAEUfpS-FB^a&n;CM*(Y4b"
        "%93RX%w$_N?NI#}#uUuR2-uIp3w%Bk-yw1B(;8XJv`g*|)J~8G#Wc>k=rfo)@L1rBasV+@e#vn_BeQ<35<`Qukq``K0Q`zJtZvx="
        "<~qnogAts8t9hWY8xEXLz;@~hd1V7;2qymF_qa+elqi;dIW0>ESz6BQ&X6O(8`+~}A!}?p=EUS0u&$F~Akv2^(iTEGgE-e7lYj4$"
        "XQ)nsuxjmPvnBjD@{3_p7Iq|BPhD4bjuY{G9~1Ic_Is`QV|(US#6{??#m8FwQr~nAdhmSI;K9Rwif3TLS&R|P*^X)4^w)4s;W+yJ"
        "!m$E@z!~CR3)i?bPshL`Q3c8z&OR*~o){oU#o?;>{BPprFx9~8Kr}ruYA34JPm|t{V#qTiY!8`uL`H<e0l9vr0`tjf{p8SGKc9*0"
        "M(k1($EmaxG53YppwJFq^sO{iDQK)As>@r~-Qknf<A^!U7M?eo%pZNSK&aoWK4<9Jpu<%SXc7!ToTw3-=qbt4E)8fmbir!{w0Jbh"
        "n4Vos`Ad94oMgYYZfL6KiUR(eFhSY-v8V8-7y+4W<wm;p1H|<@r00~SflDpAOH!%4Y&*hR8$*nf0cn=CzM6q)$+9FcK^|i$+c83z"
        "j};I?wTZgZXqq@VoW)7lMvY9wd8~$sQ64Bv;}^vU1nfY-K#I^AbKxh`EFe83)a*o>&N9bszmIfyF-UEUEkKgcrNV51h2XPbXUl9R"
        "?DxpVIIHGR4Y4<lm@;tW8LXh}ptn{|{T=pIcw^F_4K-vt&oo4sc0_rhK1Zn`wMds&CQ?-C(d7_3y5MGMMTrhVvkM%YW*0C(({D6z"
        "MHFQ?6Mqs{xR>qbI`HkujT)27WKx1ip=DzW!GPCn;qbQKL7H6~U_mE_f3w8E`DABbgQbzpfjt;!LyeTsw9>fA<RFe$o{o06E%{nq"
        "PX+`u0yk#{r5<t{^Y<M`8v{>`eBbAYTNEwk5a71qLTRR#7>;T8-CA0Ktvo;TcRH{1Xl~8PQxi8Ync$GE4xAIPGS#nDE?E{<*2s>y"
        "-&8j*pZ)bGQQeoaX|Z?Ej9Wg>!GMjk$+49J4(EHZ56c?v@+^~Og6R%$-&U$nZ8GE01(_ia?Kz2G(ESEo;2}UUZ9DXh41n6Dm%6#Q"
        "K-gh=nY&wEUgHz-z*w_rLd<50rD0aqz2>7J5)XBNi(ccA6r*SOD3hzB*#vVcw&?uK3;jeB?=_@NahQS_Gbag{T|3)b!OXUR29_c5"
        "dkT#}9qGJJ@KjEuzJovgN8~Uzj}N)LkBA@JlWQWOvno1st{Ga0zHH&0{8=XKDJ-811F}5_OcwEyx1<HE4@G*EjW<zF+RN9QcSM7l"
        "CFluPC)&E($eim~jP{QG97rH=;v^OIeD|0n-Iit;*!J*PTO%W`=@?H6wSkcnoY8!bK_=G;PA7@$QzkPoHtL;sQyy0W0LzU`Hhobf"
        "yZ~T^{mjo}GmncXxP=40p)WAO8ip8-qA$k!6BwoUZm|1HUS8f&DzCZ*COq<BmRoyLsxlZ+?Z&m(jIAPV7^lQNcr$5@L&D-O{?x^I"
        "ZZF^VRo_M<s5{uMF)Sp0E7o`>J+~2{<r3M!4<0e-CoP}&B`WK-km~HAIWl+vD6mfZHZ7j94@M7&juR%eCGe3Jh4$$5^8c|@4D6{o"
        "%=rx_;VqB-MB%@<|FsaOBLIyp5q^&#sV;a7jfVN2Mx<J~6bq<x8k|z9qWw2?_?LtZ0iQp1#S=N^N&UEm;jDLqa+vcO%pj7Lye~Ej"
        "dTiUPpNBpEiH`p~fb$4pKGJE_d@O5L{6MlhWC4Z*$hZNq3$AUGZ{r#ovR3d5TTk-rVHKDCelT(Bq|^jSy#l@R;5bjua&SmWn3kLz"
        "k&c-{A)moYXhtVSc=%=Pd7Yo0f0l@K;+NG_OKU!C#@nql^Hw_*&0B*DwEZ_0{XfN`F5)=*ywh^(^b<ZR<;KjTUoE$ZbM_0Le?GWH"
        "EXlT{2hHN!CW+i^@-<y~(2|Sj>c|2dbCIu)gu67ls%m{b;lss6@OH=ofLX*X5y^b`IR9u;aINejMPZI9K>QsfGRLN><~wuCWQA=)"
        "7fnv)>3%Hl%IaQUu2F-s6N9Fo@jVs0N9(GJ!<~JfWTkN?T`M`oma9Lr@5|zdQ{!hmX`5vLbd*K==H|ysnB1JMC*6Ge68ltuRA%vP"
        "q;U2SCvUj$NcAO`lLB>4cv7<0P?BbA2y?QXgQJbeIbFM_y3zQ$&tiPZ$Hw;Bs_G#jrIlkmQT}<0dsBdOq_(*bl95)8OYkf;6)|Zy"
        "2_6)|X7-noq4@{w{-do;Xm%qb?}-~ptGQMb?EIv(!xT)CT^$5>gxkO?7Ot{xC{ty=99e=K*0^=FmEb+PMz4Ep&t}8(N>n|JILTs_"
        "y?45XJQ1=&P1!NEz7MPVW{opSr7>Nm0d7jS7NA7n63_Nj^)bdj5jUKN>cvSyfqasL@meUpAJ>8*lPH(VL43TCi*nLuXU;f@63iT&"
        "gg<H+{^GFoe!OtJl%pq3boE5knbJoUb#&|r=P_J4k!5-tJj6LspX2xChgsiBFz8SL++e3K<$atKQ0PL&DZaA2CB_2Utg-eOPK<V_"
        "@R`aLIKE|wtkkBIl|UG6ObyH{PEZz<O@Ij_IdDBzInOZK>fqbVmEBIcdc~h<#Z?q`poNu5S<~XuZh8O?O<x*>ps_>U$W4)NcbzsJ"
        "@*&;VbRG9%fsy1E#LzkjVm6!IG!<B`blvr7A@=D*zE4G6^sPvNiT!O}VuFl<t+vDVynBR`l9or}eGemSL|^0npSkVuMiOFf#hcgP"
        "JVSnXV=o6Z*<>y4+{?jp#o}w-m%0M{OxYM%&nZ(RCN3-V&CH3|<PRV$u#TlXkYxjhja!hQX%$`kHRy9)?Z6@1_Z;vGU5L2NHx(dU"
        "x|d5(uK`)K`hxD;3ZMs!FM9)VdEUHaiB2`gLoYz10OyO&jY)9JuhrYBrp7UMvXFb(fdZ@cFjTUwNn56MyQ?}aV&z@+EThxJ^I|ov"
        "JZD<~t$&S{pA`@@4A2Ew|7box>F;i|N)5|sg9um*Bxqhs=w)AbN5hO+Qbkp6K@}2hpO>g((V&?SzbuP%f19P^;Tqv|G@OIXGjsJz"
        "lhM1m=aA7c0<MevfvAbkRT*oS?qNvouB*0-@=oq;xsX4EX87J=&PtBjRi-<pVyka)DXvCwgA^C=?{lDeAtL%>ScC{OLosktJ0t`g"
        "fW|-283hLqu|`?Y>9u$ueRv>H;!V%O7xcpIQil)5GAt)nh~;-?!@|AP?Jn8S%&dUS<HVM8pq64UAnFs92f<sFKFoM|;k9z&G-2zK"
        "Aj7kdI-$)UWVYxE%mqm|=7ct$X46AkX=TD4L#SDTF^dB0PVFhGNUq2dth1nR4OpzDTC3VF=yP#604V1agSlhfXQ^tX@*D8Nt(v4o"
        "5z*cu_Z#aI>ocr7^AG8AylP+e0gGfU35z+~lcQpxU{JSMR-VtMuo<YjMlqU4@37g5E?$l{TZtjMr}SYAXJ#<GEzO?q1`|4RC$m`~"
        "4>z+`L<^0ndwZFL@_`z`CmrfMHlyJTs#-z}j08_56*YMEiRjno5urATQfNa7R%pbT2|)L^Fr^5bF0Y<H=Tl_!&_On(^rY?A!WRQ-"
        "h}LZTy+_D{hq&dV;70iHEfqx{E-qkNF-I}-aasvJ`d7Y4@9@Mkh0ak+!%xuv@Zo5;y17Ne?bOy?g7A#xG(IfKO$glCn?O1|R2N6{"
        "b_2^2R-jdV?q5s5hkfds9w6sld)3znF0r+Os+UG(1=1O1`mH`<bBD;QGb90E$p&Q3px^`$ph=6hn<wd#Yahc2m-=cXL992hqhyEX"
        "CnUe^JSNLDC*HQmn`wKlJazm(lIDOfeUVPt!9b6eMQxAede>Cw?k25N%s@Dl8An649GfZ3ofR+(Av=Y0zO?6`|HGST9^*=lLJRuD"
        "YC65)plrUrAJ~lUm7KY(#|=T(loBuWex{k4PhQ^&EqN{h-9gu*F`V6e`NI#sOlK_wROBr<;?c=`#v=(Re4@=A?@d70UGg4k=G}2b"
        "%tSv4Pdaw%XSin|4t%eqYn3XaNmt>Dv2#632(};AAi^qU-EAGrpb&-|e;Wiby)|#IgAuaO?Sbn508mQ<1QY-O00;m803iUzq>>%S"
        "1^@ty761S%0001Ua$_%IX=ZO@VQgn*ZEayMZEs{{Y;!MdWpZa_WM5%)bY*UIX>V>WaCyyG?Qi2W82>(L|HDHOD1kKFPJqy$ND~Y="
        "?l#?RV_&AJauatoP3+Wmy6Xw?xAWLe8aqv|cM>0{G_8rB*WdH%jz*)a)3bBnbx9@5NCEi&iRNRXHRYNK4wBBPq&%YuaG`;~Z?r54"
        "6l6ms9##d(Xo<Ax@zL?oAA)fWO3Zah-T~n`s0EQUhgT=}P)YF@&5Te5_!3xI6_%CeX(RPUytz!^&FjnL_=pqD))ZddkBtn*pH4E8"
        "EvT|$@k_R>tTrxWNecF!SaVBKRY;4Lm4tD)I{oJ4>hxRu`wk7fr4(|JsgsMFpMQUOclq<(pEs9jnZsO2&<hIm9m%vYG_S3ppb5gE"
        "t;&LxOj{)iMU6%y<1qRlqy{Zywg{h6&VeZ4+>jj~<#Z0Snicu<FJtv|%@mulg6U0kd<6JGh(%82bdA2`=Z%az%9W6bqsZwY3Pp?g"
        "^qKKoJfDNX=5M$boWjKZL!ubQKLw2EjuE|iMK>-i4~?FrHR<t(RyAa#R-`y_DrmkXfu};uNKtIS`c=@>;$uX!dFWcob5_D<6L<j#"
        "df-!c)R<*Hk}yT#o)k5`lv2oObZ=-8IE=MgDky8EVL>oe9%e*miv(d<)h1&fP4-o2OT5Pvd;D<vXmo~_v?m`l<g#(zWJimTH31ZR"
        "kBOSIIo6J^-U|>gWeds%$X1GX_TvOjKdAZ^iM=9i*enz?3YAdI6wg+5LHUzjz_6ERkB-Q3nd+Q0IYK1iPc(Ae2{Nx)Mkk>R_p;H`"
        "Wi`s!%6qWc2j`E*ukP)6Vx50HgpLP<C;!Vq1wReQhp{i&tCS{J(bP`vgJb!!hty1nB8pp)MwOG3NuiV{D9o>2u9Uv3Fxb)fEP)o*"
        "@mYhOq`H<IqRw9zhK%Sa@-0q6y)lM{l@pZH!_<5ZdJ=HNu3KdVQ3`I_)_-M(!bO{I5$EX@wn)_6^^aR@!bLYNU{;81iHV76_uJbG"
        "*Fh<f9ESl>o3f-D2M#P)o>T6PhO}v{uHnsLI%S;c>C^+oAdz&BE#wG+ZF=5O>-oSDPQ2-Yl3ck)If3tchs$+%e&(H<gRRgW@4+5y"
        "VfJ!b)fJUdoNhghyE80`lJUrMG4Y*BIyaKQ=wxeR-0vmB{bU<klSw!kY#Z3yP8!p;%HfL;vP0V5)>Ll5v9?Nb)>hGMgG1k)1{w{r"
        "sRxBuZur!r^HB2I-Y#9|!=&?PfTs1WA!{Fdlr8E=)qd?!_&sj2+n09y>?#AvqYzju!<V@U8!i!Xc;t;bYt9}B%!d{j9TfGNajee|"
        "<l2l8HTjhwxu>ijpOv-PptO}ZoUriK7NR|(t-R$&$lJ!|<3z6idy%5nyF!OuJq&4squFbnyql}rYkY`0A+%DHGdlim7p+g^`W*hm"
        "rUOW8ci^=tQtTeKZF2z{YnKyKlnO5gxU%C$yui`CZS!eAy)7Ab%k<Rm;wbx>$b5ejZ&6XWVN*4@+}d2_pG3BQQ}%W4r`oqJn@_WA"
        "qL3*L@&u+{n{np3^(yF?y#fy13D!|7KBV^6<$z&)W;yz8Z-;qMo6kJhP2+>z{MN{Gt2XYu!tQOQU4zNMROt^9UE?h<8raKYV^??@"
        "SbqNEF*a^FlF-P3HP`c~6Fi8g5q!|;p-YH!+T0^?rdI{SRmTc$HoYC!o7^$o`S67PrMY0Vmm8-t6I`2nF41PY!WU-M&|*n>pv;ib"
        "cN!-9hD#4(j}^DN=H>QUVE54wx};Un?4oC$gw^ik@<v5ZP;ZW<Yet`=eyxR*I|*lRUjp_pu5g8+5$@bChDX<uZd@chcYC)ZOSZ=o"
        "H>9GD_d{MZx{}@9;y-X?y~o1!|Hi^qn}lw{MXNZ(*PCO}vhUeCofiVLFUFDTa#5<4)ZP13t!g4Eekm#8_L{Y%v@$7qCu{5}UxS^1"
        "ajFG!<$6>3qxVUQx>Btb-2gv1*~KC>dzngFQKE5)*HmIHu4&|??G}V_W&H{C7&Y^23v%uG1u5vBN!b08oxRgv8Zsfh!o~(x)<v{x"
        "(M0gJz;>}UppIX}CI;QiACT$3q^3dAd^%@E5&hG+l7JIC8Z$m0#lb>yu@)?cYjWLg4RCS&3(muAX&>fxl4&Ms;hHZ3QxUvH=CHEY"
        "3Mrmdy4S_$1uia8V@68&W;aKYh9_xQ-c!76PYbrBkvs71#j=_a8wTb(-TVZK!s9&vobE|MJ2P_|?I0+Dq6F5%EWO%qPxe=)ty|N("
        "*;6fQ)xSr4O#f~3V4y2_Uz;B{f^|7+?L)eM0Z>Z=1QY-O00;m803iS#lcIF70ssJm1^@sc0001Ua$_%IX=ZO@VQgn*ZEayMZEs{{"
        "Y;!MdZ)9a`E^v9BRoiaUKoEV8lz$lcDUpn+5Em&39x7Uj)D%P!JRoGPy<=y@de_}uw>9G9ne`=g(zF6B#qnh>XU>@!M^Ussda;7q"
        "8rkHYjD|ww-q;qLGCqNm+gedh9<=aujo?YQKvk`<LhVK8K@QeOub%c6N!=z<6wPPz8JijFfurS{-Md6<5Dv5sW`#8sWK@`3s1mD6"
        "B)QQAl`wcP1hf>=UDlXCVDBx%cB9MMNK_viUpGF6OT2o<zhmJ8jQid$7e!%|9Gy>5_{QRe@t;x1_QYCa1J3y@SHd|s?w#KU=PuDg"
        ">x3^Fh0E1^27E{?AWcb=Pt%1%Rm9*J`)aS2(L5Zk;WvwAvUOWd-f@H^1b{bPoPcFagF0snX`lda{g8|_SDun|tAXz?sGYGf2=6&J"
        "i<VZx{oH-KgePxz9!$`&qNhz@W$c49&I{j<<8%Ix-|^|kYtRPDz@rzoL=VP-Y_+H;A3$2jMZUqlJ;umZki&zVFG6cZF+|^p^ZLy;"
        "cSe$*blob{7(0oUQ(7c;v?x=OcZ*#r<Gb63h`O`r1K9(7O^gN%spE;jy;$%YE{w(l*7rm1+Wws&jqX^LP$APUN5Gvpd>wyfWk~0?"
        "A6V6E02wNyOUL1Pc3=LVB!e77-fe5e@ag;ngqCo=h7J4fNmCemBdi=+H49_8+8NTW?#T!rDtq;4GlT#_irO=-vv9M=4L{jp`^U;p"
        "8>OI|{cSWQT0Y;9C0&MP=sdBF#jA#qjn&U*q4umHCDYm>?<>+M)2dq_2bV)Yd2iPYT-?b&tm}ju2f}Wf1ZSxk2Y1zmq(lwU+-PNl"
        "1iU53%W{KuGK`PZKH-0T^gOWeVf+GgI$tyFq8$|p`-_C4U;Wo1Ll&I7B%tS|@36`ge<Fq!pU?gPP)h>@6aWAK2mk;8Apk|39p~l`"
        "005sk001EX0047xV=rQ9W^ZF*Y-eR{ZDB8MZ)9a`b1!aTbZK^FE^v9(8(nYQHu60{{sXIfkpfw@-Ly?zsJLq{Jsg^}K@;>P2n4Of"
        "-BqlW6j8Fh?e%{9%?wG2q^Pyi_T>yDyAsLae1DM}1i_E_P~C9$FMc@W>?s?n=E$mM;JY3M+UAP&c~iCrcFU{Xeh@L^`HuH29|qpw"
        "gqSrg8~CAX`@CjFUhH}D_|fA>Z}yyxofvwaA7K1$U}DINYu4U2g1vh2JZ3!?RXOH0z<l}Q6@zWq&+V({3Hzz3kF4ZHTbdg^epCrq"
        ")75#w%Y?no5BzM~_P2RoGO^ElUa}v40o03nEV*EtcGxq)>n#B;`c}X@pat*~fM5^=h-HO%5A3}__}d=nlyB;4lN|Vvm-&#ZSL}xO"
        "qVmiGI@+->U{5u1Th&~sX~VIrnw=i`F0bo+Q}Y-YB?be5q3w&k{ggBf%LQxnh65lkfZzk5q2ZC#p1uL@e%ChJY8M-CUIP1S&)VdG"
        "10vbcO!)*TX7X7>4j(^a@Hd0|$0Trf<eySfFY+Vr)nMLr^)c()A@7e_*NTiEHfjhY$nzn)zJccmIY0<*s%=~3HQ|sHM2p7c`CZ5R"
        ">HwR+?sx&Zf(XB^e&s7mf!1P}I7G958i#HiV)mMUxP*V-=0mZM*_-y7Hvp&nnRj{BKflWlUG0U6S^%Iyb|4uIS+4p{z4a|qj0UFf"
        "c$rO#tO0DssxGtlEm&K2Q;BL*)zxtH0Rob$24s}9oA<mxd!6Vj_G5Dm&_RSe?=#Z#4EE{U?ijNUDYaiB%O5`~`Ia?g`I+Q)D1Z}J"
        "Gjn71BqnEY1(wv-EM;%{5#I8Kggw3e9_*J#S5mfMRNs$v%}Nv(+KOOcmkbn-p)rARiVpDy9<*TNU$ZTUs_l;~uW$3CV50!SaMOg$"
        "6G2v6ZFgZ{YH<*;Pf`|qo?LvDoCj)u(yzP{oc%lks}KBn-?x3JeVpOe82FN#yskj9;5YIe@K+3$nWU+Nodl86X(9Lk6q9v_fh8as"
        "1Dq0+CNO;;ZUbpg*>A|$q;79{A4b0?=xH=WKwt!dQ*u8rB`-xtkieznWrj{OBz}YF(_kF7XU~EtdK=(?5?k<YViO5#=EH4Of&)z1"
        "XO}pLNU#(b_WbhOfPq$W_Vn32Sz?lm9r7cb-_sb7xycYfdU5$IUK`B7O0yb$Kn5%r2hZevRhGPwpa?x@9OVPNFDpPo&U(t8FyBlA"
        "(qiAI?p)+0Xk_Z5v#Z-pUT4$z#TUi|OGKJ6pr08<xRlgBMup8bRax~?s{z_%<!J!EK0w|plaXBiv2D15t=wF?@Zt%206?i2Lb<=S"
        "Ai+JFtU_mnUQy~Zh?W9#puOxc_KjL0O}?l>x0p1Am*kQrmmYfDykXO?Aj0t;3`lwm@G3n7Uyv2NIDi6UfQ8Ttksa*jYuvN|Zoy>E"
        "@-0rIFif#sg<S-UE=a|1L*sC&AqFjIpppyYJ;am>vBbc_OEeI3bP_^#$i=lWAoBzWJ|2h=kh5!k%p}WG6qGUi#Q3fRnpw!X3-l!N"
        "GVY*CZv2B!o?HV{CW0cvs6m??JkgOg?Lc~UW$U-}gumc4+YeHXC!^BLDps~!f)ZmVvk*I}(_TyA;bj71gP1Vz_!a}q0gHG&a3AjW"
        "Fe&Csmx7*u7{L?;<9A(M71c1Ap27|lm3D<V%TP5$64-zZ8aFaasClyk|HCpcF^kp&l&RyEZgt>eS?DZ0yNF4v6YXe>J&G2RUs0j?"
        "dNg{T_hIA^B~uCKL)Jiu3f1?nZ{LR&Vo`Ea@!OE_iJ~PCUvd`GU_2oNuB{R9j)Q|m-*_1%6dYkX4}qXdTNwzKN|0em6C;o82o<z8"
        "mL3&JaZK?o!H|v|l4kzSFePxD4k!$hprW+qhIn8a!eU#^x@&`V*B`;U8H|^CMA#o;9sW%Wk+jSR-rS4eX}~h=o78e?K~hnd@4coi"
        ")&1?oJI?|_S4F2e9LpD`T&wc#UDAw)*?&>=6>A{1#l)*dI@5Vt%3hKbGYvkuGM7X5N$3v(ml(@h4+3VW)JRj~3O9<GR42ds7;xxS"
        "Y$E>vSlWsO0OHve()dhzFfdWa`0h1bRJYmKjZGjWf<(B4aewg>T0tYWLxpmq0{u7CtG88Cwzmdw$<L|2lo-$wTjh3rJ9ffYs988K"
        "RUCZCVo|g`CGk=KhkQ7UwRWer7%>u@0rMfLm6SmJ0F-!O$u9|pC}1>H9Ci3##k-hop;8#a;H+x4L9~v^?Vk5MRI8G_Inw^uoOdXC"
        "tmZ%q<eFG5`IUppKq-06LpflA2UXwXB&N#5QMB9PklzvT5~ZI?ScK(g==@sAjf6%|0JNT0Xg@!(zG;%}*r-H2DWTYH$H8%X)*7gf"
        "Q^`yc$DB2)wV14&C+EN>U4W-FQ9muJ1^fe^j5XGYp^e|)l`*)cNG*$MH@2e)BkU!it61g`)(({06DqCrKyp(+R7O8H-G2iKA651t"
        "@Cd$)wOO+3tZ?elK@LwT;s(2L+Z-T+(m8cK*6&c7L5VpfTWvRhPP!?624z__WEt@o6t3gN)uN~4>ou5pFQlYcJ|`0}Q|-x=01WCR"
        "d^jUDIzj4^WU>f$G6b50j%<guU0rX00asI_j{Qfbt=?6-MVY!?x6o;=09JZ(BB;57)Ta1f<Wizj)<wjAz#eL?yT{bEkyh^n5On;;"
        "Vta6^S-rZhdkiKwbRJ-o=za=Gb5%FUWv>oKni=aw4$R0bG&B!Rp>Y}&Po@NB^Y0R!1B9qZRX=*$RI!0r1ruLTCoY_3#HxeAHdQWU"
        "<G_)C^N8&-h;951>>pz)JivY<6l~aouy{@5BDz{)ELWz~XAW0-wQ^FlTs)Ul=}N=EBk@=weF_})xMb2G+Z%*Hg21S$kr+c5oQ590"
        "Nu^z(a9)eK`FvDhNx42ynkhqs5|5IJPjly<){4xe%K)#Q-L`!x!pQP>^Lw0Uc`*Z-M4qR(#zVzlHehF3s3{M!3My?UPnaMMTDNr$"
        "ML4;coM|zhy&r_0K|57P%F}NfJmM+FUhs6)pUVa2RP}k9-FnCfg(e=ZnyJt;y|7d(#$vMCHL2Cgme5rw*XS<bE+NW~38wt`C}QNU"
        "nw-Dm9#wJR!@eyIPV07o2r*Hzu_iF|9x!fpfOs(;#CU)lEn;Wv0-o#q&}E0J3C}I%m8yU-g__TjZ`4+W+70q0yRqhUrd-|B{hl>X"
        "`@qgmpnTRDr5u}F;BggadpsL!b!2n9Qxncp*@}+(+}+E_q)w=K((qfT^E+_<OE={1bp8&qRKH_cjYXa^%*jYjL_pv4@V9_o1XnE3"
        "oL9Zp;CEv;iIEdJ9U><J(^(OmW{jm3>1y{C|E4w<0+guFs_dwjv()Xr8z~6{LNQqN_dNJFU>fUkYN_c)Ajgvi_QR*Y#OlBp$0+B9"
        "LCR`x_ibIvCLz?T${-Sd628=J)wGG6D#ju&!CW_t-X=*R^Y(YLyfKVi>W`vpy-Dtqi}w1C2sEmvd)6ez6Af_UHkn9<Mrp~zbPBSM"
        "4eIsV?|&q>GrG)nd)brkI1}%WTPKa$;_9C;Y+}edvdCvZ)sz*XMRj->dNJ^%?-n>u2cF={KHw**!zl&&>hhh3sY_q{TnUY~I*ej~"
        "Z(ub}quuhxmLgEtd|<~hwJ|->sFcSjCb^!;_L<9yeK-PxP~+3VDeH2W{85K7{;*c$4MKn{h+|QHg0Fvlq|iLMKT@cfe>PH_5W>kg"
        "VXk|4l)yjdY>mWyDtG{s9cSn8bmEKr^7aawHCM*rg$fjM*!1qox!3Kk8pQvhbGJ2OHCEX)*r+=g#}<z90u$%i+mev%)L5xvONZlm"
        "T9J1veT!{{$M^&$!`>#$?5tiNs06NFH?s)^*$5r-W*@3|pW;&N|H8*YMdTchiZpoNL+QyPB|pVbDdwf*J>`PIils{ag)+p7%M#p7"
        "t-cH(Nu+G-%DJ-jBv$w;9A#n2&b~44dEH-?Eb|Q7ZA#=k)x#&t>VdeoaO>Y{Pd=SS=MAGgV|L~+0^j)}OiDG4naT^K^z6=<=PcBh"
        "#7`yM3SqHBv6Fe*8z^*6t*}@w1DGMExjW%@W7n-nE7?c58T^w->ctCs^?*~OSe!rEocs8wi+}e;USf+c?BX8|Vkhs!O+ODKZ2qmC"
        "u-e@y&n(HdI^Tfed4^|Erat-lwv|3L{RrQC`uJBlFv9_a(r^%$)yQzl)%~S9v~2L0n9BXV7d59bQv{LEV`fO{y4yVAA5G`4-LE=&"
        "vi+=Xf$F=X^7vLALEsrfMweYf-|-IZ?;J!a9Mt~qDHTF7HtWYmZzT>b9!`1>kxG7375oZ2ad;iB=*EDy#+pOzr_X4c5%f%OUWON6"
        ">dSRwnNKlWf(qqFl;FW7-JZlr^TkwrGMK4O;Nc}6<MAQucz&H7_@V8O6di>IDAy3jrlW8_FX{NyTBRI2D)QU#GG-U}?=$%KG>RF$"
        "_&j>&9_q-$iMQBP4<nVFlAHmGeUCYe3}bj}+lBwabQp+ZGwk_L6=xSgJRwNsDj{_KdCb1VtuLOz(^vTBi^$E}?+1Qq0>3bUKO@r5"
        "mw*+iOD?u#2|&GMHSxl@=i3iz2066l)gogsL0?XWqt;KiBT+<kC5DjpNapvUuHkomURI#LB+M?sM>_rK@W{u2gV+A#sDX*1ZHp(O"
        "ct8nZC8U;;)Wb^1k#H9!I21($dWWE=eH9)}+hEIqI^72eWS8U{Jdnm05&MRnpIEL!zO(PdBXl4N2P~LzszO>lsqG(jiJLqCp5vEe"
        "!GR%-yhf<JJEEwuo6;-gWt`pvpWH`Erh<1_4b?%RrGaUc@`o^iRfnNUrSij5YSAH6D#gEkE{XiJ=FF$Ao|?~ux!#8IAS2HT(;Cty"
        "8zXgyxPr#H2ZseH=RDG^CP^;x0#r)YVCjWSu+0Ut+NY>{9p1m;s5&0QFxO3ff}&w+S}3R*D6uhjQjtt5Vk$qpSY+4^&XkH~dS$9!"
        "_?BdwBi3Fv_D{fqxxBDV0=l0dy_ErOXMo%Jz&Yk=Vu!H^V^fBAY4Cn_sQ?>c>yY2+cs0$^ep3fLuX4`N)m8D+!1bgD!Nzkk<5H};"
        "V46BPo2|L4cR{fbv9;fK>Iutjy*_y?8~H3m=G7Wm3C%hBeaJ2*kHL-M#Uq%aev6mkc_cu08(kZqZGXu7>x|#=VjP?)YiZni1~*;M"
        "Q3s~V?@@~7zx6fWF8Z!*AR!xzfytjq)JKMw&A9-0^wWi}kuvM|1z=kp{^mzuehoNoxA?QQ=_f%j{ibTixx=!9TuWV4$_N=9iiiHV"
        "$j`)LvU@L$&VPJ)p#SvHm%N%6e5NoBCiM_>G<mS_e^5&U1QY-O00;m803iUYS>d*S2LJ#~7XSby0001Ua$_%IX=ZO@VQgn*ZEayM"
        "ZEs{{Y;!Mga%FUKVQFq<WG--dy;xh1+%yz^uhjow%u7<HX}gG06@s*su2iX%3PB}QRgsf8GiH-G*v@nZ#DC`;UlL~~J6j~guCzN7"
        "``piWE}@jZyct?i2}62j#GaFC$E)i>ir$b`vRd?8!up!9nhl1lhq7-EMDz_;ystP>TpOu?QuZ08^!~m3_e3{HWyp@{T`S7z`|n!z"
        "_wO}Ic4S}%kBC2c3tuBb;8*SZP0bo^c(0|>nSCB;P_?2FEk{CowG@r4Si9w2$Ffe=yw%xkOJNl2HE10egztsnH>Q{`p5f*l<QbHs"
        "XzRi~E9{LmQV=70@L|^RPO3vOC|OP{oMlCMFkGJ{9wdb55XdUoH)1;~UbyE$I{wzYalUDh%|gB8{yhRedf?T@eXMx}2P(4OA3$i2"
        "4)Qm$=l=A;M7|dl&uKN*jD~0CF!0=lDk~e%#5{TGkBsOK`4ww*;IrebR#JAwR+Ra(?6H5WIj{34U&FWbxqAYMbv3pvadCj0`(qS}"
        "lNZ1KaQSFZylKUDXTBl5B(?n5qxqUhMP9u9l|YnhaBIWbwq(_{%h7e)Np3SSIXy{l^GYB&JxPgZ)(C_KH4Q;;60n8aPmVnh;L)l0"
        "OlX_xNs3naFosZOUWf(hiXte{N&`}4e|lqP@9hk3#~;W*Q1S|n-$-*M$G(24lvD}*UYcEqL``hA5rh)aa#U4ZMl|V04T5ikHW^J9"
        "#-3Pk-eo6>y<_djgGNmAR)D2e9FizraYQWZ1K&ukSd};v$+zV3Svha`))j(4b!-$Sdc*;qIy^YMB<^3DAyUX^IdS+2@o`u25ypk>"
        "nP^#w!HGDo>t!LbQoVhUw8e`>$w!WW(F8yNaWlN0Eb#i)LANE4*L5tLCr=AK4p^Ib{i%wgb{LOQK_fBHQzkVhR^<CDkcnDP`jx<C"
        "P!4&g1vs&_%@rE_vzba%#|@MyU!Ck<&+xW3kW7*?56I10xg=BhDX=l-zo?PV$q;9E<Q1khS8GU&Q+>XEtBOlM&kB3So8|ebxT6dD"
        "S`5WUscM~D#f{)aAPT12<0HBi7eqI_H(biORtK?*J747Lfhh)|;!01&2*Ns>b`EBZ;(S92lE>)@UANeBQ^3F?1Mn7^j#Pm7g?e8Q"
        "^zNc2JRlbj9(?)u&fH!e;}3n+>#=EI!h$q+;@<sy`6Fo%ORkOtI1cTor=ntk>H%tpt0n7@zSh)^^&>QUY@7-H@rM=@{|+eb-x^Rl"
        "HVjxza`G4T*+8G)o|{(8Iq0N+|Hm-#3cjbFfPTPK_%!S|ZU0By#1YC0q_8&<|6Fl1s@@%M$X_!Z<u?OYqJvxl?^|&_AZBxPqk`&L"
        "xJInQJtK=+R3=H|%id{l$A5|seO?x}RM&9l2-?(1C`1kF2y-a*pjspB)p*WZy$b>4aF4#WlLO8l8Wi2Jfj_$D2eN0asBPxMOvhck"
        "VRi<zjDt3cvksiJ7aA`Fo*9h=VqRm0VtPl9-@blXfEcx|<*0d4F$ne1;onEowQfw|HzF9Gpps1+QR0IZtsb>dDS7xELHExS?K@|o"
        "pOp02ZD306M0<FI;c~+eIP~CaB`G<`!P~6gA`l(A=ZZOxwopE%$@EB#fMs<Jy*IZzS!U<uB(YSF2>tox<qyAQctX<*99>)NxXBXa"
        "=5#d8!)G+qXIIBM*lBu_(BP9STHK)TLZ0=afmPA-jZlueAZX&HBd7VS7F(c!Q@=}Q5V(u4o+e1+RC5g^O|u=pakrCYeRrm2&;t_W"
        "o#IgAF5Wq&Tsm9gkt&3WbV7H`K<TBEOPztYIQ7W>Pw=bN-4mhq6dVmAskMN{YodixDdn}z%KZgBJH5V|R(C&vO><Xd=l)_ctdbqL"
        "Ta=;&gn@NIsrPhoLO)DuFP=NyqpD@~cXe+nUuP8#VVHT$6~V^9a|;UlT{&OVBP`T@agOHpuwm=F_ly^_pCB)>P4n;yCGhv`CY?Sf"
        "KjCW=iw?qY3eY=J!oC7s9zjkt)IUE30|O(VHvH-K7TlWbI~eDNY|@zZhlE0&DOOxkM4v@rfFn*q!gmns<GNT3S7$AJ4Ve+W;|ZW9"
        "P-|o5j~pMyn0$UYlR&ASFtA!1(s@R&hH+vgpeQVsCO5&yYTvO6tTp&(>RHSunR}y>?EzC|q0#kuPM%ymJ6a%4`nLnf+S{0UsNMKv"
        "2rm(Qd&WV^<~}jw0G=8$+x*fff&)fy9*$3=XcIRJ04}zQ)zMVt*0CVZ;cih;Y_{TZSs-hO1@3nmPJARS34Wh=d=gM64Xqjav{w4$"
        "b#{AM@A2-3-~*J4pCV_t@1ma^m<hwXJ5%{oDaU9{DXb_kaVJnq+r(v`&<)~|CoW&p49`;+B;S`oC2vEvN5OZh^ac|KnFe4@^z`<f"
        "=aww<!^R@R<_saZojH<QtH24`dTFz0vBb|YIJMZv;Ha0*GJIGoIUEY_Ll{JplM8+|yJIZ&V>-C`Xi$zPWZx&BN1e()`Vdcg+g6si"
        "GZ{WC?#9m-1joJOx%c`!A^rnUO9KQH000080000X02N1gdYc6R03;9q03-ka0CRF<FJftCZ)0I>XJu_|VJ~fOWMyn~FLGsOWpZV1"
        "V`VOId6idNZ`(E$eh-lUK<G)@6g5t=HZ$M>>);Lrns!Kny%d8$OSH{JA~ljq;uYI(-#MfxQHq*$1`J3%7oN+<!%tBZU7emUiQ(cv"
        "6cwecPaYWG?G2$?vwBk<$W|(1m<E*{X*esGPKl79Kz?O)P05B9cN-~K7DdtV(eV+l8>tL2QWg90TP6g!B|@OuR>_)d_*NFQ+M%PY"
        "kYdYsq_g<#`qR7jAJ#W1SwA*R@tO&9)3D+zmPXOSXm5NaOICeqP1Bl`+_1mS;QyAIVxN**dB+6*o0ZqBp<JyWY28$8&TB~xx~?h;"
        "(R<I%>X&YH&aB40bq2*_WeG*=gQ87S9r6dM%DflmjjVWac*al7)Q*)oH3ka6r5FO)GQ~u}uB^h(UOwjvio*r$mdVYZtT6nZ^+7dQ"
        "mAp_=>l_>?*&MJt$45m42q3orjA18?C>A1vhqhwL^7x3ruVh=2=dgjv^H{TLn-UF;T2305l6#0~lWQpcl5k<jiY(49y%+rH77XGf"
        ">%Ap|jBl;i3<7~LpdXri1k?rA_=7|>BfrzCWoxCRiX+Tm05j}>X=`Hk43OO_()l|nne%&36{o`JEDAQqtTNmlV%8RfUh=?8v)8cv"
        "C0d`ojh5v6EG5zU0v|6f{}~j!ee#mmpzX{)hZ$v9m_G)CE}$8MPBJyHLHcbFW*8Q9Ywf`n4-H7ecv3)tLnALASEtjweIPHo3?hEP"
        "WmV^UMoZ`H0~d^{cw`;A-EgW`?`Wmj)Xnr5fxl-CSv-cy({iOV3Zvxb6uML<BEwnUSJ=5&+1||vp7>6lqNbt;z@{zNEzD^RLm39c"
        "+DSj4aFX4qy%*3MUgulJN-%t&s$5-=@5VYM9DUiE)oFGLCAU*y%at~(Th+4J*unySEwp%)KuQ(}Ji4h^F5pUmxdu~9OEur#C!=rm"
        ")v5PP_G3wivav+8u9JFoIU~8d?}YwdnmLM@SDM)48g4jpJWW>G+bS1OJy*=@tDBGT5}u0$2eZp!OS(>W6W4ZZIlAi-a`IDn6E3H)"
        "!t)n4@x=oY+g<uKE-+$V>STH_RFU7YL+)olP4(UC-?4GuRV$q@j89z~3Jk<YJ32O9NPcYyY4Boq3bLKTDxOD(+ccd6*7UT0V`hV@"
        "-{V_~P{|!Pee&J7iKm!}C$;*%Pdf>GDP~$FLQurceL6=3dYKElq8lJ2N#ghcLQu6LSN=rix&~?$&tLr|yXOU5``X~Bh>E6-EIdnI"
        "cw7Q~{3ICB2LX4p;e`kID<qiskisQs_P{+!19>2&fu3%8g@hEj8UdE33#piPR1}>$-dR)8(kj<5|Ki1)t{#?OK`=8E2Xja=BV!kK"
        "XTTg)#8!MBN;=Oc<EYQ$WD%t#23D}Z|2h0Gl5XMjhm^d*m&GN#{fH~ClkhOT?EK6-2Um;Nq2o*YayoX}(FNO3J5ED38z-^P@oDxB"
        "YhYvR=8jz_LUsjI(rIKBAPCtRej^}53(c)C^9}X%uAz$7%rJ$et++O^-6I>;V-Mn;q9q4Z*(UVz443VPO1}gIGxQ7WaiSeW*cAnI"
        "aAhN96+7H;y+tC*;vo7YL7dqSpsET+$DEmuHpx&IEHOmCh!OVbI6om5c1-rVh4D~(F5APD6taeZsJ#^%1$4ZCvmo#T3O6Pg=&Rm&"
        "Pi4E~|7F-EV*Nn{!fLNH0aXeh7-26!HehZz@=>Vg&;pw7Ar!eelIuNfP{FpiFDZ&LD+-YxY=_B!Rfr(Sv5rUTW5#FqQmh0s2c^+9"
        "mUW-F00Ac*-6)-OoLWa>-0+<`@0S0!TRX!n85)p^U1;lg04tCWIBv;v=pGefn2ovHQ9}V~X~VTFD^%V<UG_CqcR71xMQfgD!3WHL"
        "P)h>@6aWAK2mk;8ApnCIu8%zi006rX001HY0047xV=rQ9W^ZF*Y-eR{ZDB8MZ)9a`b1!mlb#!TNXD)Dgl~>zt+eQ$5=T{8Whe|Ci"
        "-x4EG;v#XIv_&or?9?xTK`+UXv=GT<cbB%6qW|8RC6_nJNo&Bi#GRe%nKQ%5Wb(5rEIwE$g|^a4RmfrmqDbi%WR_LNYGLIaLW+q>"
        "QG>y3dGPic&{&xZt28X>_d+gJBC=mG&&50lKCRIpR90rB6Ah#a3S}Wvf)s<3f><P*SQQ9#UFKzmITn^qHI~XqQoMykBx{@pS8P&P"
        "OXuXEw@hmBA<3!~0gJ{r%a21yv@)i<5xRhK63O5m<!WusJeW)-L9o;+hg4XRWWpHINRB9NyGONz-Ikn|=I|Gp*bx3GS+&SQ_*9Ye"
        "LC~C%<H<S*f|s2R_ZKczWih>^=q`j!){6k>hrFOCnzB_`GYI}5!#nafW?3K!DiEeF*TQV`9Icj#5B$Iv2Olu0*sr+@k1aGJku1u}"
        "Mlv;v+Qs=NER50vnKollF3e_+U}NX2%Dl`sm9<uljVQxN%6w8k_l0lZ96k_Gb}pq@+VwYY1wQ|j+6L{e6uPazM~{62rYg%!QlhQe"
        "U;);tGA`|<R#Y20`VUEe$w-t5RcWo#CM!#VbCy&YK{WC?;%16Acaafy1TZsoqWnX~^3p~b77LKn7Uyt$cKYl|*b1z$=$}44KIzVi"
        "DvvU;MQ!}?>FAV9u2H1z`LphrpGT?8{n5$Ov+gk7Qh!Ayc}{Q?olGq;gM*)3cYG$+2w_R>Zf_Rt<q5M*&Nv>DN!Qs#E2_x0)UU=="
        "Ik#L@5V7bSmQ|MZ1ncPLL96?9%><^Ug0)OjD&QlG0B;YkL(c_dE+L8&4gp=Pjs&xeXpKou-J#~x%xm`g-oSbrYe^!lPQE}9n6=Q3"
        "L@7=py;h&&A0!1tjtmD;W!&;4)#usJ5vAwIL={W9s<iX17w#*)GF@WFfbpDe@UM#Ufk01Jb`4XUujT}#4O8yv$)-=W4)10*A1G}>"
        "_O4c-mi!3G9cly4Go*64ghQGh%u=;}g5rK{eOP{|@|dbb3MuZwaE<A!hlMZi_;v@|?*0b12X`W?&>?qpsRQ$GH6G|QImXp086;fL"
        "!G#@wU+@Db%rmbUQqz-q=)E~UJACu;_-xiI%_Z7O7o+N2HAB<LFFa%lcrPztO0FBIXi!>0gsGNG`2e@KQ6Ask&gz-v4(huDeKZQH"
        "#5Q3h<dRq+151)};}B$(6&YM#zIqQtE)I703Xm|e9cc0xF{G0nhcBpA<0pIIzP9G({bp`I%ViIK@B)q)Uk!Hn>o6BuNDBFyOZHl8"
        "rKgjtAx;hC)D94H`xFJPTwly5v%WndhiQkWaDa}mEqgew=L}w)f774Asf+a;TOq(#1r2^GW6w;7sJdbvBAmfC4%cNah=J}|u!mS("
        ">$p!YHfC4&VGtjE^#RhyQG}h!@8Hk<<QWTslq_)3RL%L_^>y954AKU9UD5s6Vv>%WVRCQUtNo9@?6bz_x;_IpM*;3vp2wtlkGVsb"
        "9v_`NA%*;J){pk1pU;LSxLxc<#A-OK(I)$xS=R9-wGL(m@7JiY3p%b>+Z7Ws5AHT_kMRE;>5fJh?IU&`jo7vIo<z};gMB92i)}YY"
        "Wx~^Y+G;u+YnnN;NAn}@xv>?_70_AG)9fhZ?*YmugL{7C$>YgltT7zk8^<u%nB4t6QI2%<7AzMcUGg6vk?dusrB=E%NLqLGpRewF"
        "8hP1^Whyl$^^O|x%yk=a$1}vi?JgJg`r%F{==bLOTW`<m@4AB>>%Gt$UR=6ec4y234NRHR0wigu)4oz7fh_cX^rx90DS}3Pn$lWx"
        "=&nk;1&+aQE2WV7o@umW89tuF@3i6+BFA1-Cebn`VDoIFfU!usotmfNbCC1`rm&u4sWc+akap+F&_H9Y<VCBF2905VXy3Y*MXK(Z"
        "xQwLDWc-@Be{?m7t&>SoX4+3g2r0jqoY%*(QuMZ07vjABH~U)o+gIwSA()7WHnKvm(00XVa^8KD>R{m5j07Hf@NfWdubqQeo^YUI"
        "xOrc(8sn^Wdm$-QZR$M>@-*-i*>EV=v-W%5PhR_KnKB*vKt3-*ShQZ>?C|@pbQj{yxG+5vVP8Y=FHlPZ1QY-O00;m803iUz0#RGu"
        "6aWB7RR91a0001Ua$_%IX=ZO@VQgn*ZEayMa%FIDb7^#Ma(Q2KbaiBTE^v9(JZp2~Hj>{{mH)uwCo+-b$7GU;_q27J#IrYbmzk;Y"
        "Wb3LN7YiamiDQaX2-3DP<^O)&4S)nekdmG3ZJkmRizK?yX!ILR`M&=yP4leaUd4B%$m+7%d!lZVy|;rqtYURm7Bg?n3ogHVtVp~9"
        "zTR=~$M?T@A}*_}SbIPI_nU?9`(J+X<rmp@S5~#hsx=@GyuGnuVv}d9_WF}3i}s=v?N_mf?`c(Ty|~PC9^<saQxAVEo1*5G8n9zE"
        "qSO!H!zEHc&sJGl#w=g+?UpU#vPiSF9(Vaa?=L^R`T6qg$LQ_fe!jfI1=bON*zqdc@}eHWP**IjTS`6vf3oWGfo*p=pW&rq;One^"
        "eO0iX*pyDJ39BV){!nYgF)PX<gCUVnkP)(O6x`T4g(3Lz3lBb5?|%94<7M>W@*i)m-n@HzHIujhp4FP9D}dtaw%l+Yb?D$L>~5l-"
        "SDYo0D4QyVG3C8zZ@0s1CT=4n5}(84aFpb_yyZpqyObzJbY24fJJ^9V<4L4O;7;Rz2~WM28#Dz<Pk{S>LE9dB^S?THej$m(NxdaW"
        "+ydtj1i7FkA?pKP#AO0<y!4wooxk$KP_M3Ts=|R-u~;L&_K^?%%}iabnBY-dZns&DH|qvuB<PDs@YPo-{0Y6ZtUUbYWrc#qE8*WL"
        "8p4hR_f=MNd+Bn{P%V`fh8tPbi|uWaRe`(^%a2t9a>*YuQAg!1U4{mX#7*2FJ|)d|CjvwbK!s>PG?<99Z26ky0uW_I0%(`#q4%}t"
        "|62Gs%3`96@VrO4lrZsU65nl%?=L=;Sz#^*1=Fz_y$0F~7=v4(NgJ+fuq{d}l5EXIZ3?AY)_qoQB%S4s7lB``{Lo_pEMk%5{Gy{+"
        "iJP*N3zP$mRgRr`zSywym*4u~V#6QQGlP_h|JG!%xPsqDWmE4O7d=@jsUdcXAS9#CioMhd;ADlY4?~kOmI>~m++GsUWmT0`;J>3O"
        "JQxmk8Z8jZQK|MG+BBAd-DY0Z6o2vJO<4j0<^f@zBy3>{NXqMY(`h|yj^IF=*ztG<7BSv%Eizw$I^i(2bBadrJ@0n)9QJL?Ylc#L"
        "(jt?{J&?Wq4CV(L<88}?=iWWf)*GO{yf4ryNC$@4$ZTzz^%{0PWw5oIrgbu#6t5x;gbP5BEi1AVHluH5ItB(veGo;or%|Y++&l4n"
        "xzvu_NTXUHBG~8&Gu4VBQH{dOQx1q@vL@HazC{9pobc)fWIT{}2o2r=K`?K4960J--<TVZUNNqD0_)OrUAL1gh>ua0+!*Gtm4&6;"
        "I9y}J?tS<>1VrzZ1BV0u1T2UjMS4v=R7V>F&4z+)VT4o<<b+;DYg<vuw;=k)O7SB#87QaS>OaGewfVttK{GXX#qTl!?rqr((YLe!"
        ")BnT%B-W*JT#_s)2&;X~MPQACjnqa%+6OY+JsaWu+fp&)Vz@4;-v3~E!<A-#MB+S>O+xK$GqGi$M+@JYJLq=&gcctvb1W$(LOMln"
        "!;Gc~AVh3UKguDjv>%8CcBFQFs=Xqt7{M2o;Sk2+CT$7<ch5*t6>1!Zaj2;oned9T_TH8SmxEf+nHK@wBhgD>BAJ|2MW{XNLBlB="
        "kL2ScA{J>|0S^HyOIvYK=65`3qnd$}kbl&3fngz_up>oD&^oIn<*=SSFYf_OeP33$OM2S7rT3};++yZ!vIJrV)QwLZBk3jBnwk2i"
        "oH=j>BPn>KB7IG>;2@5m3A_#rbB96510ZDNvU$;LF;C%v9jr>q@mtTUfe;Y}@2`qqfGEl{h}t|ro`bKJq=)SQdT1%o3TC_QZq?P^"
        "8i0W@&^{I_t9>SFX)QzL#_@}s73+qrIcl=M+t(ZL4oB%S<P(t`PZQe;jNyRd`!!93-)`C^>t~T`?d}O%5P{hdNU4N%Oic4ZoKQ2c"
        "kYGM>8*!_TnkIp;8WJTiH3kO9B$B^Ml7#~`It8>`ut^T1j~~mr1}&d3a{Lb;{!e#|xIa{C4{=mKRCA#RB+}atA3GrQQcxd;Nl|~b"
        "V3YOgEKKO^9lS7+qtdsiXw4Q5il=u~b_bgP0HZmigR6*bHa@-MQD<_8&sKRE--6|(nU`oKq{W0(%zOe#l>?aN8SI-_Y7i1#braIJ"
        "A;6YBL@H~Swq8i%r|*%pTX42#m$AumEZq<El?B%h!dM-^mA!`;tKD*pv;u{{6)X#lkx7p40nOo9{%5xk%N+Qk1t<M*fYVqrLpDH+"
        "+hdS2g1hjpKl?}@d`lNZsOJ;^i<)k!?~gLSCcC01Aq=j4eVpDikoIf8(>TE5-@q1hoIgQBU~}FwN90fzYzLWOg88V!7%P1;Vh)F%"
        "s!=Y2Ty#2OS@Oo35{1r|HjX2bB(UwYHO}QfzFvA~W*!mqUF`*({hoR6uGE*05bc2=(-mdL9KTtMR#Af{_z7YO0SUQzkP8_wj0`lr"
        "$OSs$1%#tP@Ak~N4X(yzP2`-M;ogH}*Hu8P39m0+o}N0T&M9<NnWIUyi+I;YuyAvIuJ+Zeue25+@N4jy1xaD<qhv{(AhGnom-(L0"
        "@4)ydxs+qWnUd2k*p|{_Y>9J9iA%|nPgV&6V2xHsfNdB`3H8$4IA5*|7&p0agNr^c3bzaXn73Js0>^PDjX=&Ev8&3ML#%HN0nNoL"
        "C2l<k=|e6+#p62AGT$2k%2s1!kHKp@QirnDwU&#Ug@Dingy;g8auz~bBeQo|c#;!aGk;Q*a9pY!VLtD99vCv|j{M&mR<WYa3a%21"
        "NL7_C>_BHE1yM<8FS6;83vsjxzgx`xDp?nhn{XgKSwdxD0PsQW>MTYt1bG~&eZG&dh~wJ}XWOhGpa>xvWRdjY#yShMgiW49YQn0h"
        "iyT>#0K-g-t>F}LxkJhFfBE?O6=hvCeI$=RAWh|QL$huf#yK*l0`}w)BuDWrC`H5y;1v)Dfso8T^-Pbp-Wob&9KA$RN2DW}DUH#E"
        "&nhc=wOaYqS>JRd;G1U43RKrLdl+QnBX#>yo)$$SZBHvWunz-#8Rtzx4-YE3iugzH8@iPC)Kb8|Bwu&r&_>4XOnlpX*C^L9Rcjti"
        "R}G81;;GEb0qPp9*LGL?Sk|NKRkL320ZC3WX*ztCg7FyxGr~^?sEot_)-Vfafh6Qmeq$sOljkR-bSpr%*|O^P#FWdq%5rClkeO#6"
        "qWGHX#SZL(Vo^9V$S}U5Bd;1VqZqcm&K|sa!#(00ie%1F(&VXBU3vn(i4>kd7L_)IghK%Vp^0hK!U*p0G!U@%^}$-EBSP`ZJc(4S"
        "iiS(Bz!4?*C~3w(+Ev#$L`X3oT2PZaBRG#8E1DI$9SWjaK3W6q09wlQ(l<BNL~d-3Dw6a~g2g|QSgM^s@MVWm{wiCeKz0(eWAUxM"
        "6#skn&DnR~o~PfvV6WoU`HR)+<*T!^)pswFmyE5>zWs)$ar)xBR|7!AXy2Slb(~quIO|Y;aB&sdJ#u-&75@C@<A5wOwtDZhVg2e?"
        "JMZ|~2wZ6lmfP*DP@aN~0)o#6ugL3W_axwJ0vxOcZL__Y5!u)^D{$A(sWecPp(`qwZeUn4fUx(d%-%_(@<NW}tsC>4q+hPxmF7)k"
        "fJdjKqjdnonL$i4LWH$_F7tL|O~9%Lj8*^~*}vrij3R)vrH?k5k8UCBXXbLn#^b4U+02FdjB_McR&I3&R?T!Wkw`f=(ZV$vRCW;7"
        "_I0UP&wS65G8V04wm=RhT}md5WafHc<y0U<C#6rx9tarXo4Sn49LZLJdXlB7iqRsLFrcsc3!X~X5Ln{Dw1CiTwFjNdb9_QOFE%{!"
        "-qRvqEWTfyy`WAf4ptg?y(iU<a|Q?!lLuO|#v=63Y8wU&I*Qqkt}s5y*t#eM=1F!CuYAG!_iU#_>ygO3L(;Z4Y5Z|j3W08wjt<1E"
        "XL)~P@~%dIR8ai?BD`-;4b*|~Qd+X#n0ewdY%A&uBXXXO-?zXtVMO))BzjN)?8b>E1u3BE7wg|G=StNZ3+GgufFYSAos&*gZ>_Z?"
        "*$-rJ`*{x2I(xu17yAGdbBpOvw<bKc0M_2)IUprWdPv}x6^p?K&2VGX2|INVz*Xdj_QC$}@aThvnuCC40$Us^URN1vo|Z2<%!CIx"
        "^Oj+MR;MVvbMSFqTKZ$L6AVo~d06=;@cN0)spt5f=bXlz<3GncnWv}I?no`||I!_mpucfOoNE1&Y@o9SZc+APwExmYC14y`Tt_52"
        ")Oay)woj`0J=ryBW=*W2=a9%h9#Kd_xoKqfto*ZdZ`$!lJnFcb@*s%c8-T1p33TlZ%xRO?o!=?Tw&g)D!)|(LYcOt|BNS6&9k!g~"
        "Sk$Ga<S}HL5IryqTeEJ0t9Gl_<B-Y0L8=VOqctS+vOCX)1q)>z3=K`(Rf9Ft9>rkD1jnXok5G5@*4CV52{mTW8IpPOabq?|8YnyR"
        "QED)a0t_Lw8&QOlh+!Y0N#swu6l)@Fsafr@DWG>mFr*BN|0+e5iSw+xA^WoQZAe<fk#+RyvPRe2=V?wzv3hKofD9m_LvP-qnPk9w"
        "V|!HFq}LR;MR{MyqR5+{I)1b#F(ei8D}+iaKgY@lF4#9&etgU)RnK^!^-Ik>l+A-)_NNw^6u1mKLaE!m_1ZHj_t>O|1$DN{>2(B_"
        "RXrEmo)!(L40K<Ssc}QXRWFoodUcXMXmEunp%M6kLtT0$1eD~=j4(iv)uYsKc*ew5_1<lSZ!f$%lCIkse4)xPp;z6CxRzUL|4{}|"
        "<^&y!uzgsE;xvG2&RxBk<f)Zfg9WOaK$}4vH^}xw+oT$1OLh(mFwD8M&~@3G)UGk%a8#!2!;&B_VzvV<)HdDd2sIf#-4BD99O5t~"
        "$DJSnO34cxrOR^NU`G#450)$eDW}BJGb2_FP3@Tbe0%5>M_Ov^IfkPqI&3=HUF|n`v~Sb_<0{+Leuyr`cNL5kv7J^g@hn_AGW7`d"
        "z;%MI&1P2|lX2J55=#}0xfUtg5Gi<S&*{tt#r_l5YGq48Bp`h&5*28xn^PpKrGcrS?%$4Sdd<aMR+R<jyuQ8mVV>3*y^#ngP=CAN"
        "M90zn#I#k$u}FKe&Adm=H3Xg;EFj-lNB33$sY+C{!3aJ1C*qh6H@ipIW<-87fq=@`7YDP4$3FGDrrMQ4)=AB`MM%R*&#I`4i|F83"
        "1`+hkcN-AZO3^7$q~46{-WI_>hkenL{7@H5+kWUMALQ^qe1CyKVZ)^t;`^Dpbo(N{A9^F55U7_1kI4zLLiflUZyM_Il}1BWZwv$N"
        "5Jkw;+5)bk|1KPUs3?>pj8-Conpq=6+Lj4L1YC$(>bewaayQW;n)mVbpGa160)Jt@-Eq7Obb%tEud|fiM$|9d5Sn)T-8D8Ms=HQW"
        "B@;jxK&}dO9J+osWR<~8ql;d>U?RI7b|uJ^dY}qz#sjA^C}tbO%)q223k*N(OgD%W_`r@?PEsv(WY&T8?WcNX9Hslt%cJ_v+wjF%"
        "a+&T(KceBB8oTbNQh-08bw+&|eS?`1;0N!`uu*tN;^qc&wV0y#3~$>C*=mH&!PMEQ)92G5RsV1Y1&(^}NOvz*u+0)%|K13YmiJ>@"
        "f;C)S)*uvGjZA2L?}&zCGkbn=i=@^0ILLU%+8m{0#40V;#IU$dZ$FKUrj5CfGBYv+B`GqRC(GR^vR2X+!QGAHn=JJQcPT3x<EWO)"
        "7rSLgLieJ#-H>pIQ?eCTmMRC8MLHI@ZaP|J2#0E4pJ^%AuNDV`9@I%)8RQ$L+Pd-zhoUOhSa%Pr2?@*6rQ()OE{{-DrB};IrGJi6"
        "K2ITgS^WV;bRCmh3P@XH_FKD-!h6c6K)QP_2${dOgJpHKKkjj`raw7mbyfLBju~7PokOPy)6v$%c>-|`c+&+Owz_IPF;bS6xIr%x"
        "i$Q2`q|ezJ1ZG`vp6nr;0FQ_+7P~}ItVfQIbr6lGPRNqGC02X8_v9-<Eq(e&HWq)Y)(ysV@99a9aG9QC3K~U88Ap+h44W06PSG&z"
        "><3n@<ds0EdP%m071Zi3k8>0A4O`j9N*U>j;3aV2)7zGF)#ch7Ov9KWF!1N|R<q}d1zx@1@umJrfZ?PnfO5>Lp7{NNgBg_tp5bwS"
        "7IU$bzsO+sn1_?o53pi{cQo-sn%c6Mf#LwnI;&kg14&_^&~z2}QW3-nzNl%`-VUIC?xqY7NkQn|$OTM{v}Ebi3uO^Frh^+sRhqea"
        "jr3U<sA+vJo7SheB}a%8qmzt4mBlzPi{$n5nWa3@T%O@6G?YPUD!)1X{@Vjc#(`g)9};19KLLJrdU|>Q0VI=iWZeXoR|<tuH6K;v"
        "c{j)(fE`xrKzC_iADBLQr^d|@)t*3poOVwjc(O)K;5t^$CQv*{)Y>MbiLL52_1bXHWal{ilesyL>R4uuV|WrTW#`zCkFuU<4jO8g"
        "E~y@*brfEQC=M?(0z-xBxomeHwVPs(r6gV24<|87V~(mgZ%gizHr<Uz7hLC9aD4{6s^05GrFqFrQ+YZGe5}mqAknek`jKx9{R6u`"
        "ilVFX8P;+5yJt_rsdAsWQG)ve1Qt`s4neEp-7}$|Y)VEkJ;@FnWbA|sJ^TLr5Ee*WAz-?fs}oK%FV7BQnOMmo&|_5uv$ZY<p+AU+"
        "ilujIO#eqTbPB3c6Y7WT<r{Xb)BVdY!;IJ}ko!N#`sm-WK03cV{{jCEj_LFojwmI{ANnb=6i&x}vPk=%ucMFe{_FCsJUn%^gN6T;"
        "^?dUK{`(iNhv%-{)Onn>kuY?dsUs{|a<P7=W#;;gl9{6wF7*E0J3H!EMcx}{sH-MfW4gO-MD4A(M=3&f#|J-xH27JgBNyBcqK2&m"
        "6R&C>JA)BXkKlLpkRQS>gBSbfv-&-NCXPX$#PE1fKA-rZ0OwBwC^NzzvN|}Qd8yx3P}NVFrYy^Oa^Za@(8pifAHz^Tw|6SbEwIXn"
        "DrZqdD~uv6zC@8;q?DAay{P&25@(~@3`Rcx3s6e~1QY-O00;m803iUi`j2nk1polq5C8xi0001Ua$_%IX=ZO@VQgn*ZEayMb7Ns<"
        "W^Zg{E^v93S8Z?GHW2<EApe2Tz9fKOx^8QS76FztZjm-gv)SD+41+<FCz}f;Y9wXHE%M)YM@p9B*oo!>jYv^^?)AC5BuTD$29gPr"
        "J8w(jmC4ym8*%Fx&n1JqLMx#>Q@Mw^<?+2KbIGj&H%OA?@Zj(uwI*YddoDD04jfZiVXSYq=+NQ8FPrrde(@FZlZh?i`0ya?v9nTX"
        "c}C!G4i6aqlzhWbthjKV`_i>HOlbwQHj2^~#??FY@9fLP<@xmN^7Fe-V|K!h;<*N=+Mr+Z?#1^>k0mI%h73?$NlP?Ls^)`YO^dO*"
        ">t{xSX08CCQYE56$Wmef;<hkoCstUqgq-IBl6e1d?<wH(eZbyw;a!FO8=!8=8@ukIS2!K+u^IPb5#S38QbqS|zTA(Fm@ju=l6!{2"
        "Xw4C)!o9KJ%`N2WD~5>&j(XaDYOGNCJmP~(o>TCLePGqq!HhMa%5#L8aR<YT39G?v3`j)Ixc3O&(Y8O*orUHC7Dh|38)z<h;qlrE"
        "c4E28Nhf;gz_a>&fJKK|%?rj$seDJ|pORi{{Xm4)ew4-eAVzC{VgdT-a1?>1fT3?Ot+-ZmGLARQST$EU*U>=Cq$vvR>MNd;?bSYq"
        "xD~ie`9)=wOCu0s{Ku#tc3D9V?@MLjVXQq7YF95_(@i=0cnMSVT(~%-ksHR{3hX*8m_>t|tiJ*^UwD>*oukGr{&^<at|*;aYYQn@"
        "pk!?CR`~JuDMl2Up^m}Uv5z64ok@Mh?|SURucsj#xLao#c&h{@`#;ebPuL!X+LdpW5RlD4N{qca7RQ_0SH@y)bBiQRf#4QMyBPYr"
        "Awrh5)ZGAV_Uc$iCs4#_l>;Nf&SrWY6q|8zyW&<pYvWyMRiR-Td4-YsAgj-+bgsS1i-M93t!vl^$TH3}Ut_j)%B?HPt$py}Plfuv"
        ")VjJJ)yEgM@;H+amJi~p8}zIF(1O@;!<$A{_0)k*hr${xai4w<X0{tLXW^Sp(372Y!dJXCG_w$gOQ@shCgiHg(Y3BDz%G@eN+0CF"
        "&#l2@-Ed#-P<ewJ9-t`~CRG}7Zbt7X%oJW_MBYK``V*#&DLO*`@Dy|r!0@R8KNY@2y2>CU%Mv73GIabJx41(FC9`!4u}O?@Qota~"
        "PWOo*mGg=UW)te4t8W6ht!pMKO`TCE@QY3qrt)fcI0tk9-^E`Oj?wP>DHanp$N%J-Maqo{fEv6-0jXI-c|$0&0&?2#T1j3%l&Kw@"
        "+f}ZK`fV}C*xM18hSJ}UjU972FWka-YC^G@jD`5$63Gy$I)|s|_WZ@q``a5_-uapdTlUd#r>NRhC)i419a=rt+>9thpY;_ixh{i;"
        "s**P?)_(EWUl==mmDfJH31^rS#)5r|tbx+a@a{#%?Jb*QG%~zb$=HSdFUL)`+LL{b+XQ;!ys1kcbt_k)5tMyl7S#8*_-%|dpmL#0"
        "8Jep|yOEWs#iJ6`v1>Wq%8K2&**d^6FZ__m=22Q0qp7Sqk}1?Z8fijVSb{c4mi_Q?WU5ZmjUqemBaBu6U=2r|o4y{MdXASNMWETZ"
        "vaC7Vi7iIN&mCOZ5}}zE+@n;)UN6L0d!+a{{(OCjb<8$({i*aDM_X)}pqY?bjI9-SeaBO14^WOT0sY2f8-nv;oZzNse@-vXagO^I"
        "Ce#6qhvA8)g6avAr#<xD$2l<0m+xa6eQi7%U9Lyg8heDRi^gDK@3<O2akxFrLx+$@r}mgMqKqf&DHzAG965@!l+P>;g!sypLZFA!"
        "QZrlT^^i<gN=VssN*&U4dPIFeFUmcwMqNYz&y@}f)HF$IVdxiCJYj8f{8fMdcXN&|9@P7IzJA;~Q7aSa<AkLN?zmyJ--j<3=c7R;"
        "zx!5yJ<w(a_UQONY28DO#nJI#qf_&WerdA#`u<JrwrWhjXUYZB)h!sv0s9u!zh}wzMb|{h&O_nwqQ~O5g~EXO4^T@31QY-O00;m8"
        "03iU|PzfTA0002J0000b0001Ua$_%IX=ZO@VQgn*ZEayMbaG*7ZfS03FJE72ZfSI1UoLQYZOt(b!Y~Yl;k{DtV0ot(K-~}*fDNI@"
        "b(3O9;>t;pxIF=hjqMGe{)dPMG0$~(I-Wst^%jVI*+bZ3glZ&N@&e{lTLlPa<t_OJ=PzH3s%VBODnAB9?7O}TrU@nq$`lu$5)Hr5"
        "pwqp*$<r5*w5nBDsq)RSX(_ICP1o0_;T*;29E@<mKLoFR_Xbc)0|XQR000O8001EX$w;}{{ssU502TlMC;$Keb8=%ZVrgb?V_|G("
        "Wo>O?FLZKYX>Mt5XD?%Qa$$63WM6Y;bS`jtjaW;M+cprsFR=fCRSuCCj*~4~6maTO+W<NA0kr7FF$7wo)<h&yASq{^#s2rs@F9_s"
        "yp4dw66e9;aOUI3&9dxwYRVI<NZmKhn@-Ax82OnA{*@Xo1)-uMrEI%~8TN*CvOJN$Km0>#so-C)NJqKSZ_O7;j#RUT3#K<&mfham"
        "-te}Q%8-_t)9p>IWDCmX)bJzmlE19P%Z|L3C2by9+tN)%4c*APMt_B_O|WPrt%|Z&)Uc|+VAn(&dTdxxbMe4br?@az9VQElJ)W83"
        "H3yeogQ+DSMo}(cnx(*MZi;C5Me|NJyu1X<Qi_^C^okX3yl7>`U|kifY2TP4m=&FpHE)=H(YJyv(%eXOiK7e4Dk8CBrBoHB|Mmvv"
        "A|1#mL=mzZT;{rC<(go!i;9C4TCWF}swgSc&uV*nL*Rj}|3wPjw~WxHArR?TCdkKMHQ|E5d`!J3LK01(qQLo@DE3Cfs{tXAun%=3"
        "zmQ7FuEXWnN<zO%e$g%nvYS^yo<_HFHVtpN(V&mgeTHjSCOhmi8@t9gr;;6<K?Ny+2r>6%(Ne){rcIb{UL!U$o52XHIiIG*hr)zc"
        "f|uBH?!-3QmnIqPUz!9TUX$2h6tG4!lK&wETNC`>+xQ>Zo36LHdU|#v*x+_!)`T0@+Wo^#*x0W;?000mx9SJ*u|JGL^aFNiZY?7_"
        "e4zRbd&E?L(&mAgJo7`@3QHnS)vd?u)vH(JqtHh6CARV>@dU>E*L)m$eR%!^<EZ!ndS*+U2I7wuxz8hm=ok!S=`5@CGo-tMD6e<Y"
        "LZnZ3cQDVI3`^RE^**oX+P2T;!(r>^l2Cb9Z3<i7J6GiU%#UV=>%Jxy2=0X!Wrw-d)oL38!Kw@1!T~XQT)&w2w5~R^>zJs9;;+)G"
        "-(z)cumq?mS-~q?c{pg1CvkNk-;JPvdk5+W2TnSvP=G;L!@soY*>6fom1iFusFm`B*vcy=3^x~2o+una6{&hSgdUPpbd&jDqt<^K"
        "r^z5xI3=MxvVdxM8e3>X7%$HES(K{Gjz<R`7`oNOV{vjb_SgZkw9M+-Efch5v4%&&t>3L!6q2fkdE%_anj<h#t(kTAG?d*o0YB)9"
        "PzKusZ=Yu?^}xNI;P}$S)g&!*axiJrr=>oGD%g_!T!K&j7#G=^Jci=OY~W|;BsfoN{uo9LRRC`IwOaOz)%rM~<XmT=`<5Wu7kWVH"
        ")aZHqT>0!hr*%0gAt=*@3<hQIT`|+EvBfe=9AJCKk&2!1JJ^k!16fn3c&Kx~H;w?gck65&j_30_KCaej?Edr4|5~o9clZADh%~p*"
        "P$R<-Ez7>`8{46~9|muag*L3yJ9}1oUuW7B3=ZeZ?#FlU{Gg_1HXBq7xle%%Ep`Ch?EyrIQ_WblyZ>pJ0-6)f^QJk{@^h@$vKCJn"
        "#}0!a;#+?TMpQ4}dfte%%dMRpFM&nA2XtcP=T5@@k@NRN_Z>EoCIAfJ7U-7?d?Z`|sou-a<Q`ArF@a$u&R-Z4(zrbfAFyc=pw7Yc"
        "qk!D<*&XB9SI{p5Zgve70*a*XfXFlCDL%M6d%f6{j5m41L~iF^k+&p|;=M~k4Rpy(f%I_V9F{&*0pZ%xC3lhy!$Xd|6IlH>tx|Rd"
        ";1vLpxXB#a4$G`*%bQEe3j<6RQZ$!{s9==1;{ks{q*Kb5=}X%(qj;H(HueZG9O~DDSD54r;JS8}867QCw&*V~yX$w9e6@FL@vyag"
        "$;o}my0MUh+2zJ+>5cCLR@YFl&@AC{6Xj~0W~w}d7@Zo>G-KqU%K)_-{%{ItvO=i(M%Q6ENyDzvCJyeA@V+#C@a6Dt#E%pv;*_~;"
        "!0j3mWhmZvLxKRJG}L`q$#5rB<!M}93*49=d4o(QB_|a0;5aD(<rz{G0g7jq+`pDcN<ulJs+$rID=JWw>KO?ay3*vU^qhxa?76OT"
        "`@Sg(zr3PIH5C;ac@;>(z$@kHcw&JGJxZjf7&U5J{*oS{LpPpy1;o$d%MiDhZCtHvB9-+OFK4Lo<BLv4jj4~&x=WFG*+L|#lO+&A"
        ")J}n90F&rc!Y83>uHPX7_Y#Y+7$?N+3gP%4mOEO_gonxIdpxo|XsIK<q}v_N2M~`Ru}XYufKvP!tKmV8H3>f{tHLw2!B(IAbtpot"
        "nCsvZ^BsIf&V!G={X6m#mu}L%(fG(^+%Xl2JUenf(Rf6Y7ITttrYi}VWPdlyeMG0m*GL{QtMN4k?ucYX?~%rUm*FLuJz-kuXB-9c"
        "vuIQ8c3>rkD%?$&=PKUyeKv)3OQ)xUsYzfW_V#D@CXf<qJRvHnZD4Z6u<LHZl(KlT?u>44{sT};0|XQR000O8001EXg{`{f9svLV"
        "<pBTyDF6Tfb8=%ZVrgb?V_|G(Wo>O?FLZKYX>Mt5XD?)Fb98BJY+-b1Z*DGdd2NxgP6IIvhWC^19jr{P1Xl*8Lt=uC4A32=Ip-Qk"
        ";=^$y`u6yG1r@28Y{&8c`HwMXcX!_dO%f>E2+;~M;P-?a5@vaTOadWI#DW751LIc>tnvN`g9c9EZM*A?F>TW}OD71iJIuLfO0W-}"
        "oQ)%<WV_<5_Y{ZoP}riC%J@R^igq|YGZW7*+h#zo3PnOxD#d$`RLZJI9TZHHF-Ykbscwh45Z{2HbqD0Dw`;h4{8#AP2J|TUw$2Rw"
        "ym$;Q2-q0LJR7h{wtSY2U{Xs?q+s*#s{@YGR7GpfQOy$axHvJ-Wg7}!-EXxtoW#|Ca|7lHeaJcoKB{i)gBpCf`^A^Q9z7|B`WJC7"
        "sOTS0$Q~K9!|n{RZkr!aO9KQH000080000X0Bmf7a~KW)0JARu04V?f0CRF<FJftCZ)0I>XJu_|VJ~!YVQFq@Zf7rTWo>VAd0%jH"
        "Z(?OGaCx;FU2ohr@;yNQ1FJrq<g69j&7}voTfk|X!vRejG`Ze`VF+A_yXILdDI#TiTjamr%#akRk6qinI{}GT;>>WqXE;(6MIW}j"
        "ELq)Xxs|_jE$iy#Z@*r=V@+GH#DZ0|X8Ky-ZPDep%s#yR$aKqPCEE1)v**v=)rC;!tgXAM$N)=LtZrF-E83Db*{0>qwMyAXAX0Lz"
        "Max>D0b4QLcnip-`U~Sg`<AwU{)`zMwi4^Q6>P=xn>*eX3%2SsD{7&LtGwiL%ksL)yS5dTX5atwhYNNi+Dep6+>5+JYQSL>MaZVy"
        "HgyZa$gK!(>o&h0-lbK=IP6hP9;fTBA|3D&A>JaL=g)u<rPzDa*H87Y7e9ewcpC#WD5fh#at;!~RfsjqGFeHTWw8?F8sy<hoWn+`"
        "!CA1DS2yQOf-p;V`r2XQrMgD6agz2F62B(bgvh>KvXcM>FEEo5>^EL^;>WhF+c-iN4fwWG8g$H>T1im8NTYxzGMFNNzzP1DSxH;*"
        "sgtcxXyYP_-||w4F$vmfNgIw<RL)CTZD8>%Jz?xiaEIoa<1IUygD>lrXBV&z`g+e>z7^n0@laf2D6wtyE=CeY#i^{awcxq~UKe18"
        "TbYX`E!+vYxz?$91WS6?h@~O2E^Ds8It%0(i$)B+c??6pi?&uV?O1m@;KlIxzu*XXK*va{x`lu?4rW!hU`iLulk@~py)Qy+VN_XH"
        "yhRsLp~pd7YBa~&v@{;w|1hwIpEy~B<{8|*3HXDU@7l^ck0p|@7iObW*N!g9oPZOAZel_W`G$yJ%MIu+qZE}<Dgua@QptiHEx@Bo"
        "neWa)?iK`ykeO%_T<rKeR>)jiks*i-8YLIVVShNhXx)`%^f^$?G^##YCL9eHTSx-4A(H6hix?e8k40QA?S2-P7rfD;0G2PW#*8kV"
        "YyoRvRaSv3E^rsfJ}5~kl{VT*ajD46l8TdLlrDg_RFIlAuW}I^u!T=h3FB2k<kMC)rPOhhf`U$y%j454(%Fz4UePDC)4!v|%+k&N"
        "rQ7{W_3YB;^C@V<u`pytA5eQ*2^9o0+KdQOYz1WsaIEGwLT9R58J)xd<s=}zn3MshLv+4mW)o}QlsAp2ikJv`s4v-t+HE1Rv@(bM"
        "C^yv>E1iVuB<>*8Y9_VuwN`Aw15I7#E3mt+>(moMjR{k}muwRX2L3Io9DcQ#2Lg@!TvuXPV*=0ckfC>sEZ~90zW%Z89Bt1qczENo"
        "D7mq}1Hp_D`bXfq6bSBqxPy-rYG2$I2P63BI^f0c!|nvMAdrbb5-^hsOPkIDe~(O1M&~v<1NV&3_^?2=a3*{ibU=a>9wHgfXkD-Z"
        "zV($5sAoCzECZzKwz1bu)on!y3Y#dk$4=&Hu6CL>)}m5%Yi{ES(r2HWhk$gWDA7XJwc1xgpbtWDxfVv(8u}^dUCb?VX;6tpL*zV+"
        "iTB<*5QeO)<fl$R2MI-1QL`#xjYHA=hSe){K_&|b8>x2?Ll85CD7r?uo)=UBkp(45r9*cwtOSXx2*~=#$MI3hiz4>WqaZiZy;xdF"
        "dSe{92jE&5)R_WS;-PyHuGenQuvKt>Ki08B53S+3qB1+I7+w|Fy!T{>KoI*{3;<Gkl$yF7cBBYSPf2ZifQBiB))%&6`e;;XU=y!_"
        "P19vtv8~%(Y&xWwDsCv4We1gFOToT?fXlmLP%o{jki@8UgVRZ!1LuWN;Bpk723d*r_7KFn6=MG?*?0ePBlu1B2sQ}rFfUe>$j+h<"
        "IHM`lND4~Lu)@h|T|1=>d}*YhF;_!UU?0!7Bew{ZKHznV>;PtP1C5e2ekIf<{SMY<RT=Kf@3TiRKBwt<Dtjadd>P8)T9&?lh00ZT"
        "jVa0s4=y^t#xueZ_V1!m%QNcf#4;Klt)W(zP<@F6WIeGw700g<9}k$k3{h-EOHtV^5{w7DwL2w@VjOm2-?Gz#wemyPH06$giy1aS"
        "FRj^w8bq#<BWOwxAe<}^`M!n%Z;0dj?O=j{ZTCpb!wonNA^NWS@F>FV-<Lw8d4(OwWn-%3ftF*`Y7o4U-QlRK@dmOW>QE#aQ-*0r"
        "M3^wqZ%_BbP6k{v9wQ{qs)OBn<6@m1t|SlYvBdTg&K$BT1a_s^a#mnVJXX&&Nm$IiT8+bFlP&J;u|r*+Tn!YN*!<i}$E#i3*oVlv"
        "{WYW8L-si=+C@KZ@{S@DwFcm+)Ho<0W%uso@#PRr7^WMHds<B|3cmeUOa_5yY?p(8Tp-JW=hsU!Avx>-W`ir=3ytItL^KK@gTy|j"
        "Ec;aPmNOME$X9At<=1UpVdpVO`Xe|@W;&zfXVV+jbT!1W)4JtMk+z+m=B+u(;tVc@oUzVwh*X@KSdbDrIH7{WuBv#Wu4_Gb*5$cS"
        "_=NQ7&JnxODHwANstvmCjfg3+jZY{CTA!KIq;Gq0o<q@(;g#t+!{}HBry6&;6o+;grL(yWI^A?eR5Go}%!&5`TNRMg-{nlHP<T$%"
        "hLNi&eLGeK@@A(L)oqkO_oM;C!wj~KQo?})C^i^`YaUdV`dG#8+LLLZuP7+BDN|Co8FbzPa6yy@LrHh_Ug3_-^Lzdgq1Ev?1b?8Q"
        "fCFv>+(wW`GduqBMfB)s>bRe$glE{9YGIEOICyufnNuOkjcS|+wmE-Q*JW&CQK~i0_ypNtGP{{-`&Wo9h_wEITN(8Idc}(t-N-#Y"
        "03>?t6G|a8q_{XqVTWxzH|juhBj~~^Y2lo02#SZ{8PWb!5bRNrb@wF;!)&h=!eDWZk5jfi#tFcmY@>Hl!HH<|vU;4Yf~}rtvJmrO"
        "%i)FAGg#9cY75^Wm7;CNwJiX7*ie{XCDdk0SWSwzqvpu8W3&)N(|qc}RZr2H5BsS(6K;|eml9j5ZCo4N!wK#obM|CrZR*-QNVCM&"
        "Cgm$=Yf>ENB<wqO^4~IqEs$^EygqP@f<<kXtMIrlnmPYM-{^^r1|;KDy*DZQ=>`GpNyy36`~Ghg{1MF*UXN|4Xbv)JBJmo{V;&G|"
        "3iyP%j6->FHe<zq)O-f7An2f3>`F|oqn80_Y2D#CF?xUT;lp@-A5ktg4I#h;=hQhsfWSZFCF=5^QKhM=D6N;NF<7rCWM<mO58L$#"
        "qux>NfQ?>Q(b%=d1~>i#GFz<Odn}q~_PSP|C-Q`j4-oy-^}x7qKM4$(A(`P!cFYb<%+gsmpX}n(`B>s<sF4nuPzO&q$Bo*q1B}cH"
        "**y8&INsoa5qR^YGyH_Q5_w$}M0I))V#n^FG)6NO9AkrijF-*%3zp4~@iJd)^7m;@ov`OpHrm(y9s>olfy}(KMsQ4^V@UsnMuu*c"
        "nJ+X{M&ECU0<b8~0<^&V{;Y?jCu3WaYwTGWELN7|Q{dx@!=E_XjgxJTXa3Q6+Z=32g3*RwvhRUe;Hti>l=D!Woo)FJq{8u;z=_0`"
        "k=@uYzr6bf$o2YwerL^B&_CgzG9A)BJT!2ZiGqXTx=AHeHDE={95M@NTxH;&8=Nb(VlD48NYxvC4W)se;*1GDX)v!8b$tVYvw#pO"
        "3Y>@F0Ln5GMbTRc{RsE;ZIm6GZ7e}1?s=}u9mp@>2o3#9SE1OTCXOHa@NH8<V@qH8SiJ}N2=mX7(Zn^2vI<&RjMxwgXP|>sA~Y0h"
        "FC~0-il5^Q;ql4oSEnampG}~>?uMK8kLeBj7l42AdV(A%*G`!a&cS2_wOkh2#H=eEoAP$&nm}<6d5}${cmc8<kF<461>3Xh$5DpJ"
        "w}UT}raVn~>52_`WSuGeLJb@<vzEvA)BOmSBnKX2-8J-H-uoFlVsUJ?e_<!2!QwGHP1wtqlZ7O?JSU{9!TXQ+D6qvj7z6rKi?%?c"
        "1mOZ)TD3rAN%Kx$;gG(6v~XY79nr51N=$ET#;>!$seggfrX8)r$$%=xH&7w!hLy-r%dFL;Fm*>?bL|i)b#TZOvg}&nLj+%O+|eX1"
        "e1ZK&uIrqae;3;=KmPIT4gL$8pAo>TH~8-Wz}Z6puclj=l~;uPkB{?9ufnCzg8@0I+6}~U^i2<vej09WUEFv8j7+PPM}<yc%S_xu"
        "0hjm+S2?%D=b&8r?j`_D2D}mwaDt~jzo&6W*%HfVZk>|PZJ2MI46k1y@3;8hS7%?p{+ns<O;;kr4&A)`m#t|_Wbu?t5sK^GEqITE"
        "Jo3wvD>$RB@DH+md1`J?%`NtoE8>DL;D?<Mm}tQNFHg^}9vI+<Y~TbxlU3egrR=}TsPEJ{sr?dz=qeZ*yYHfKW3u_EPr!cWe!*h<"
        "EzE*+C5p`Ez69|EzPrbXKCwMMcUwrzMbm0)$wDF;(uD_qJv}3rz)zB>^HX<HU&p<_M~V49WQaKGkcYSZ5scDIY`%Z$C37N;a6ota"
        "$Eh<PcHtFy=Z8Ly_Spw$zcGStP(ymP`5llPTYK<^QhT)F^L^ORICQ+6Vn2Xp^oL*U(2sp!CIJT#0Yyq@_>J=xzth2H8y%Zmi>l`N"
        "%xnrT{D7Bsx^#zRs`ciaJdZ*yad|bv+W7D^In=v~6AsPhKAy8+F?t5JA5%*EnVzAyf>6B~)YP!}sJw1y4CFfj>ib;o>EDvu7f`5a"
        "GcouC7M6N9s1?XEnh*Yjk3rDPz4Eiz91NCWQJtP%=rw`++LYQ3l{GlRhwQ`e<+j_}IR$Q$V15fq%n{uLSsyQYIbN2CaOs@vDp23j"
        "sgF_;K+*dvr0%QTe*jQR0|XQR000O8001EXcP<LOpaTE^#s~laCjbBdb8=%ZVrgb?V_|G(Wo>O?FLZKYX>Mt5XD@GJYGq?|X?A6E"
        "E^v8$Rl#oKHW0lV@IOp-k~XqpXD@|Y_0UBE6iACb1bZ<IgONrS5sFkvDvtB@9g37JIqCXhTby|_^X5G&ilV3WW(!R4qr?(gW{Y~I"
        "q&UO3`wYG*A(&MfOwB86K*<7s1EGw8rZX0#vi=Czk#U(tQM6nv7rbeewm^wRJ^z<UnN^)EELV~VU<Mvs!g5h!<%=H*P4EYlR;%_L"
        "FBd=`p^n^|ZNPX!X_QVuuxAt>p3)#n<AKW(-?qS|g&bC!^?JETVD-l=c<a%56$)gs<PH3i!&ji0J^c9W&ONhTD=-exre-bfws-r)"
        "orfcz)E^}Q-+eUW2H^)29o}oLbR6BkwOCl<+@rzn&-5|<u?HqgT8Bnfp2xL#)IMaNMakH<<NsMH5r?|2kLmnL#^gCpW{sfjv>XOD"
        "T@xqp1=c<w96n^9E#b0=tP+aZILS2jgT|{(3Nunxn|(4YXcCpo*OO2G_#(+g4n^I`7c38gOJw>}UDUiRQKoQ0eyr_KFoiFv&psIb"
        "j``;9*A$GZY{A<6A0_dsi=d7`5^4C6DtQysv<4Cp6|uY^!3LDu(w<wq4brEDS@nW4w87;urLA;?JuEkKGdb{kR@A;V;D+cfRMU1&"
        "1YE*1D_%}amt;E^KrjS7N}*vEH77y5kEkBf_ELtE>O|=c$zBm2|6aM?p}|J!a|YivR{Sjx2F4PatKX1oU2R(<G^#|AjU@OVl~zsg"
        "OlLVQv#7CnX_fO;F!triFeR@h^uzD4an|vrFQ#!?ox0_xX8B9L(&(kXdOUwUHsQ`Z0zK4d5ucQQBE)N_e6&=b`@jAfrWyPLH(~qJ"
        "BD?ke;;q1WKBP{)Gvmg2XV+KXhYd4*L~A~sV+{16y>Gmx+yEVt8BR)<^yN{eqlAkmC%=1-%~O&~B=HiI?_}#JayFVk&EULTER%J("
        "DdBcVcU*;^A1J-Pb>?@?FhEM-W!Jwz@`&h<l3{kMEID-KSK{gLj)WMy<leBk@2p4sdKc!a-0O|7t@D9q@`&+zC`Pw`*Pw6!7eP-c"
        "q$oFW?3cUqINN}CGfCQCE~x<hv*0=PzH4p{N6kv2hdm1{Np|kyiAOd4+i}uGALDE>s&Vo>x!PF2ZfgJH;5dkfu245x3Z21&Mz<0s"
        "ZT`T7p|h3%QNPlOJRbVd^C#Wmq-#>?wLHdRL7shqNy@N3|9#`*XRjrHa(lg-<>Eh3O9KQH000080000X0IuxscDNV-0Gw_B04M+e"
        "0CRF<FJftCZ)0I>XJu_|VJ~!YVQFq@Zf7rVW@c<@Ze?F{W^^uadF?%GZ`??7-wpUb@VGb}DUT@0*}Wt>PXOPp7cp+_TWGxrg2{lO"
        "W_mO{aY%;b=y8Vrd#bA6%_fH<dvOR5A~|dNQQci#?<#V`F#L11FV}1x{8U}N4SxCY)7fgvR=0JPmtC-BMa`Nv*i=o>ZCS9(I_O;o"
        "AF3j|4$7)y*Hv}fCNE#SeDQXj)g5b2tFm|uG>kUL+IGKVG{4Tv5~kM;>zXXbKig{GtQZVxSdiUiIlwI#!24JQS3k{yUA1OKQ17pc"
        "e3f;1RnCGHGzIK|t@ijkz@4#OT{T$|WL?KfYF%Zp0{|0-A?_{50Ub=OAJva-)^3aZTK)c8Tb1fZla=dg_wvQ2sdhn~bvQB*eLurb"
        "(X-#@YuSu{e`_q(Rn1C{YqJf5t+f^4QgmZeTGg!V4@qdM9(c#nO<pjBO+&BqO|{C38@AhJ$vW$@WK|VK2D9XxD^_QD^Sx;GO(Xp0"
        "<xNsofZ9hnF?#tTfaiLj7wc4xNpG5JU$-;fSJhp<%m2<A-kk2TGT*Sa6W#k7cEnN+HZ54z5QqjE5kicoc==+@HUXX~UFSD2H$oK0"
        "mmC6b3g+^Ll6ITD{_c-a%)9UNZcBH-Cnbo(C~U677$_}hyM|>u0SO-@-PLwq-UfLY<UpWNk?pS6*`*ko!2hhHi}Tmt1}6dj6VHO{"
        "FpLcldQ%BMT|@xIet?$uO(_R#*@HNeane=22V^o9-DSm|NznNw$lJVZyR2M^t{JDA<P#Bsxd*WdjY`(61Nx5Q1OQ94c%iNHRp){7"
        "`IbFi>Rka=b}=Ce+GmKQ9U_abg}>S;ei~V&$Xnkk3(o=;?By69_I1HLK&d@C1lhhCYSOns4ApMBR0}CV_9SgwG8eyLdmR&hH~SI@"
        "SN5!`9Z+I^ofmocDB4bBpIRb?O@jEGhb9}QQiRfe?k+;)5OL_Hs*nkY(#3)?5Nlf%cT8S7j!Aa6&;`uUfb47MpBh9Tdw@Ght6Tah"
        "25<5bWaOL!J_|OX-rVVDI^N<BH$`=w6=~bC`VY&?^FOYiLUl<rKOx9K5a!indCP(yft}w`=ewq<nrIV#uN4WO^(VNI%^vRhCaCUM"
        "bKk(FT?U^C@KYEoMBgyI%_vOZXkDHar(y7lOIe_N;4^c)K!Rw33E2*$DM;IS@L6xPgx1TTcK8RRCn$}ZeYI~T04X?d!Yrtfu81}g"
        "Y*?onzBvI<RKo@R&pwA+0^yjgd7oe%CC4K@eTG6P;wL@}!i=lXHZhbz7^_1)7FD*kPZTOL4Go4?jR{?Aieey&<C!r}$#xnMURw0c"
        "N1!B1oAL%axm3Dg9scH1A`4vo*lnv)jONM{hkHh@_Up`oiVZgCG%JcIMtmA0@-|iW0d%qgNN%y5St#bi5z&#r)+|uS<rYW7mq_YP"
        "(i_}FhDy{yEN555s>{OWX$*dmI9ynbOY7{Wq(IOf$)RrtPnsgZwFnxtGg}*^6Wd1zWc4;0%;&+FxOp8KOf1o`2pt9AhD$)rkk&V7"
        "jrJzy2;6GYmg5sNv%)wb-cLLB&_&pitoOURjpQak34u*ocJtSujadmMACT%i+;^MPe^K|AHMgwsF}P`1$PHkRhq;lnBR}8GJR~#g"
        "9mKi>ecY>eZ-4l&181fEPdO>vEf;c1`X^l42fq_y5>CUV^U#Q%wEhp&iCj>da=ObJWZQtXtE>j|925^25e-`bYdt2Gh=#QggApxk"
        "D_j(6vRP_f*crfhhT=8IdR*rs;j|z}TA2LXs9}36E*;X{qR?rhNGLCx4$#u%WYD{4g+J0BB;>z<Dz!`id<a5|i4-J!AegLtj0}b_"
        "ct&iVSO#Zt01N!Rv^YnC<6!zG!2prv-~^R;g9RKbAp0Epz)3_)EnkzL7i>V*vvs1mogZ)j&#rhmjA}rRngk>tNr6u%?$OKG(QHS-"
        "Jg5-EySydG`7$8khz>A!x}>oX+`UHAh}`763+~~NyWFt59Ly5sX(ypxtbdM$qhOYF6tijS4Y3Ds`|}+LkTc|}v#z>jCGzB%;!Xfu"
        "Zt@#-xWtfFU#5wAH`7x{bVtW(B*&o}J-<FBO$HiW!`Vp}pNYHco!d4#g{e8!iu4$d1{3#mKj%t$>#nu1oGk6-M7??1Yzs1fvjGH#"
        "J=>F;SolNno<|<-vN|m)9<iB+V3$JMJ<D&lU7KQj$aG7-m|oD{TZ>zU9g4jgc1*ZjgeED3ON-bB3CRRJPI5MB`3cbA+=Ha}3$E-^"
        "p}duVqB*O5Lkgagqq-?^CR@1UQ?QaBNZ<mnKxe{s0n6yI=OPi)+?iSyX!-AXZJ}s3aBy;RzymQa{ds86RJyA67+Q45lqk?70msV`"
        "_fFhQM*+@);lgaEW-!?0Z{8m+T|T679CtZt%dEU%k!gIyF)B5n*x)>PGnebU;kf3rM>+I;NqW#=3&1{V%SH$y4l%Lq!=C413kj=h"
        "s(`@JeMv|9NyRTCwDzPd@O!W`AX&J$`>MH}ha@1@px9Q7iw26@p~7a3tp(6{;qA#Sdju&-NXJ$~6dohi#qKc7)uSCr=Z3J>tH>yr"
        "vlx1)6ps?SL@%}lc4QIRPW<;43~8}-KDbV(%?0M&y%j{EX@)gHO6JKt0Jj`SM3)t5k=~w<JtIDKf|2voJSM+ee$Lo2+(|v8M(DvO"
        "l4Hwdw;~IgeN_lT4i}+?R)!YTP$K}b)NG-;Rn%c=lNfTOXAr+F9em*Anv&G%VQ7@(zW?=yw}RTa&=PdlT7+jT)%GGGzWq7w77W#%"
        "7Q=WaPDWK(R4urg<X(YQM^Xd?^2sc?MjGP2lGq;%xC7Cig3c|jv(@eVV^y*l$$<H}JhU=N=fakeBGz!^65K(gF1X7IjBA4h2`gVe"
        "raa{X0M3*1Spfg{S(`fc4h%%w9gc4M+$Y(PeAMtV2(?dvI7JB_&b;GA2#3|aZah*d5tyz4Gcfr4@JKp73`Ihdjj=QBsb|!8jcKY+"
        "<Z3z{xn-o00I~L>#+Cb+oY)_3MNU|7c$Wh1D`WbyKbtyXww-y3Vv5kWane;0r?R*|_nQC}eenK=GuZGAob{T)dCVkVTLqc$2u&Z6"
        "FPfwFsGDlNU*$Y`*gM(?VVTb;k(rN&;LOKMe6(2~={YmU$j}-pQzmJ?#t7l!Au(ErA|6D}Yu7Fh0V6jV3|r2p0FJhs$mYDP_Z=?8"
        "b=TtJAiy>A0+h|i1LWk#15_F+#*iWoy2!3s(W2x&y*NLQ6L6>Pptt)x`MW$6TJ1a7E|B*CO>nUXD8J8|HJ7=xC-UpIS(|lWBMB)S"
        "n!`d7$)-Lrr@w@!DQjB>T*)_<ASmPmv^h~~dszTG!G$U!E=uxt1A-fLr~xDP2?l()j=KW=mG7jM`-lMmgn(rN>r(&|APh^f$%m`H"
        "`Qhv8Peqt128ZV6X>|?<=^GsA6TU`&K1tx<)m#Dc4MA$cF#|AUl>#{S4~Yy?^WX&U!Wi7Xye+HyGK?MG^N^WRfhC5hC@U?Nh+}D%"
        "$OiO<FF2KiyZiy%EiOs=P(gS#j{sEZSTrWUBFOfI>!R8l0Tmod?4Vjhy&Sgtnl*QD?Q6|#*e>-0{`4HpbAz<6g<fGfej8$VfBogi"
        "4T(u=b9@azBiBFu30POc=}9O?&k*1w?jDl<8-v!<FyMi&hol7u2-kX!ej;CGZYyim@%tcp{T&dup}T3O2l4+1r3PyErePq*zM@|9"
        "np*hJaTBLlUVaeGzNDZ53Fu29K+dF;ptL0YMLhX&>c(8nB`o6Sswc{?6sDPI*yE8j)3qyv(w{Fm>iqw^6y<GQ<WwXDGXu?w0)`Uu"
        "7_}}sCwJ;aoSEPB$(H^oL(&XgVyI*t3~~sVhhTlaf!N=jC6mtZw@2uBodwVSr_VOdb!UpKVYh%TTsiJ{s9Lm;7k+y$G~wTn9`@AB"
        "Apf@Bm(2J)$Nd7|L9!#KIDT~l`1J{pLt@~W(u1a{T^)4ngvZWDo277FT5UE(UXs`D$%+zN%S0!;JGu~QmpxGIubqRjCE8eBt+vo0"
        "ergq?nzGeuzuOl)j{mJ_v}*e1ui5E$q64hj-M$vf$i`KX*J)WbyZJ?;7A92RvF369T8^Xg7`KhF_59+W<zXp7mSXgKZb)6Jg3LBF"
        "&s22Y>`(z=9yT|-e8CT$29`ygVE)<-O}1h<0k0C<M@lU_RxmIg8&z%jN%Z`XwcG0oe1ch!)io>EDQ#wkM<yOm+cK-$ZPm$DT6Tlk"
        "IT_Kw#0=~#0dC*$C>2x1P$C>e@egpm$~Bs?M#e=qg59S?fx+WD&?ky~W`b($0Fp^3Gvj(^#<j-5o8Ur21!-mx62`~H^3$T@60H~z"
        "?%?0njE9r9!vQi94?IJeW=V~-9JZK53p+d?UHc;tB!DZ~uI?WB0%r&rjXe>whXrcb%^8QQ^`xho&)Tg)-2{$G`(&{lwO~$%TG3yi"
        "YPZ8hc(yzWugr3eAS*DS`-qv`4o-qa*j>KKR#>RF+Ex|7E7^T;_3Ov<<J%AK5L;L8e);voyY$DazkcxKqK_&^wDP43o+&kQqzAI2"
        "OAC2=@c8vrzLcj$IAEe#&ian+098U<fR~42!9=?ybI}YsIkCmC3*wTh(7}CWHU7aDN~|%m*w&L49nFef=vqxj&S7i-cW!K4XVqk+"
        "n_7~$>umR*R*7QK%q>opsoZqUR@vj6yhV)Hfjd6`F$0@e3Lf~=SWqQsy~X7c;m*e6sT-E5#%Wvg-ki-V?mar3<m|I|t?oJ8`MsKk"
        "%i-*@$Uqw)48!7BB6%GuBO=!!h(6E7BI&_LQ{FMKGM;D|(JABDPR?A>KM^YLIBB<guxpq|dkW=+`HD*_PNzaAyNHrv45(wo-#jgQ"
        "DYLdREz3K%IaZY+`WH)qly8E!MWL!`bg&p)q7_@S7BGhUZ-I!YyiL`AqRX9UH)Yj!`KnD6FBy{AtU3_6=Fz$~`IwemkDi@&V>zd1"
        "j3D)GxRZLNZ({|mS`b<$lo?wzaG=t@)?iQBqrD}g2-2E3S_|3Rqgou*DKADEFSvB%^5MG6qPn4iZC#8m>5UdD!UH5C=MQ5d43odX"
        "MVPy6;^0*f{&yLYLTpeWisZhe_2m@on8AS9RVSozg$6CM)+AhtcQui5px8N}4tsb35F6{F3cDh~2kG=e(;P{hq<gF>B1Q>Mv(m>%"
        "lL8plr8UcLQ=vO?!PZVP75LIom&)9f%rn;8ti?+u%8@z20Vmj_Ih_IXz8N3!%&91HP+O9M7uMM2T|{t9TLKJAESf6#24~jx^(eV}"
        "n!XU8wq4O=r-qP(Xs{C?bg|QSZf`HeDz~1DAH6Tc>!+>dX1!$6td|X%`O`W6!0N3!z@T^CRQLT8gwv|>X(%ZiAWATPL09NWToEmS"
        "C7q%m5C(%HH&osd0Mee5GsyIrV4USbxw|XC`(tbAs$sCK@9hPru2<-QIXU58rQ6zbG9?rF4GvukX!@#38q=><2*?@Uj9r;FnuvAc"
        "OkybmfQX-n4#`p#*pmi7x;A`DJfcQOE8b*C8p_A@`SP3KmwjDhj3EG&pTa0u#;`2-k^f{iiCe+U^bWnyyS#OB)A`0Jpm0&4MTFca"
        "AMcz-7Gm&b8RanJ!UV|%J{qz!rJ^t5CEQnH5?=YGUpw%I$moQr6V3H<4h_dxpEDSmyh9z`gQM*h-z3)`o8<C&0G*s1n)&(3yK8N<"
        "x|;o9{1Bh}E!rer{rX0W8wwdXQtXWg-t^1ZT<P|2JRfCOM*1>LuAH2kF`oVcavZ?-Zbd%V;IrKht|s<#$R9DEzHg#sy4aL<R-9uG"
        "?n9xgmyp9Nm=c_m&n{mz6<5VrpM@cP`HO6|nX=d9gX))N>53myg?KA`zTlomE(HwfoF~U^Fbj9F%ii(_ii&!}MM{@1=(83?4by@M"
        "VSy(81Ak?O{-l&VpX^!9M0&z?+hD%ex-U6+p!Ug`PPR@e3nH-U;b2h&oj?>jmG;{Gw2w;k@inGYnPOn9m*T^vbj;-*@K)XTYCCkz"
        "C;Y#tO>4=ZOEFC9mV6WHSXNRlXC{D(JQn;W(>6tBY5lb}=ZUZ6?(>tcQPtUKtBDsjpw{enxG+2=@PznTz^^J@)qBpUZA=Dg32qIB"
        "$2{A{O8F5Ew+rNFt_1b5qcqypRz?f#`qeiyWoiYE$AC~+_BZ3B4GKECVDw?TLe0~|(R?PHkBt6WRd8Rm0`99;zYSEr(WQ<})A2;X"
        "XWj~gbPUe4`X0`<M`$~Ih^RhmB2nBbDCFt?%kcWocp09`hz*wE83K0ElaHid{f?z|!-wxuOW2Xid=}N1SPD*ph)cFJla%8)k?Z>)"
        "Ob9cXk6D@jSgy8Bh3`QQi&_iAJx%j`6muCs#~!VN^q4AA-{RT^m;vm0tg!!6#gDd4VOBsMTkGf}T5l<%h1Bu-#N&z@C)6mqtY_$2"
        "U2o&lsqX2}bjz8D4c7_&)G1jbS4*i_>NNzNV@(Swr$LG&Jig816MZ*vLZAHL#B+lys=33jN!eoXL*C+tWzNdJI8?BDGLb3)WIC}d"
        "lYN&LZ9+u?IEg9+QY#s%3J8W1iv|XKhsDWW0Y5I><soR#y@SZeB}RktGQv`uAo7L?UIiC1u)JCJKwM|lK@h`TB<H<D9nLGO9aZw>"
        "Y15CeUJM-&xzq_fCOmwrNaRGF)Q7*0OQ}$BJOz*ZvmLSIq^ezrW8XB>uba=jF&XEgTz=I?nlg=p&|ZPQs007H9NU{ZMTwf924eMa"
        "?__uG2z`2tF17TiaZ;s7hxBL<D>hsdsY)KqY+u?9;rO_x7vIw5Qv)%1BNAlK^d>FvIU;dE!)3PI4rdLH{90)ag2-dLqBK*x_P&uv"
        "c^{#?BoekToW!1E><kr>8lJuQ%NxEmXzbvI(^^rVy?4OS@ET*~$?_R2Q&vE}v|L%^=1a<!@fj#dcS@5T6U<wmOwa~H`qEM{Q<UB-"
        "oRYipZ2Txq27J1(MKXyWlF19DYaA13NTc*%wHny71uH?NTsW9f_}LBoCPRMBrKA?Rm>Uo&3N=pR3+U<f$;m%C8Xrq$fCXhlZqW#{"
        "-{i5Gr2lxhVqtN5@voOllrIa9KftHk9m}4q1S!>Jyll);1D;p_2C~ep$3l$RZ6}y4!PjJU&g6883O_=l1DXsg9LTl0pRC@8zUuIM"
        "qrR421@dQQj`6BFcCnCsNreUO`{N_gmH|PbOVr)Rs4_PyhzFtQ0*Zk#PT4lz#=uB=k$s|09YfLtBV;&I(l5q-HTQw`+9HadWX8eH"
        "uH!@RimSpS{%zbW^prSxKbEW4qfg^f-I7(Dg!GHD<3263Qg}Y1Ua3#MvNGTIMG4X6Il(a9>ss=!ME5T1T2)NLkt<Kh)Ziy&-Diwx"
        "hAIM$({)2^fNA)FuCcN~HRh-)LHYPn4Z6ro5~+i((Mcv(9LZWo)S-R-6o;|y*lSK|;&TcY)Z{+8hAV=+l|b?t5ARQ8CN0t?=nfzG"
        "_jSb-9WkaG4(o&|y1;wJXAtPf=b3yod+F|UvPgS|ID4)r6G;>|_A%pf$PqbHq|2y@b+l0VVu3Os4+dnxv*f_DWx&_r|EEEW`(Ejl"
        "r|kB)R`~s2Km7dDyHD?aa1VFf2a##Ur0B);nyC;!Mv+$gE){QPYTk868f0aC=ebv^Y1Ayss<$ci(ZlPdlfEuZlfFi^+84jAaxAHa"
        "suJ~!DpnPkzWlW=HuXnf)O0Vy^c&sjjz@j{$i20R9`!VDz8^L4k`8SJYu=l?gFBJKev^&JfQxt&ds6U4)C7}DVpNUxs)hS?)K!_o"
        "8GRE)SF-JP_QWSwkS#*gQTWv~pGXrFau3&M(1x7pl5tPi{KE*;bIi@Yq_8eX+G}w4w)oj*{FVhBoLsg39PhW@hdyU|)TiGKdh>+Z"
        "{1p<;n<e-d(CiP+Qv8JBYt-X!{fHOyC%o=MJ2<jdS}J*erL^(Y0!<5JUoJRS&iV_4osSIAxgJ)m?t(vOEqh0QqPd>6*lzx}a_9?d"
        "ypkM0ql)oyyLaEPQN!T<$4~FBu73R)O#|mY>;g;8g1@jwdPC+tYJg^6ccSnn{PFGkpFHM-M6|iHuV8Xuay3Qyhe8zo!Og`<O5aaT"
        "Q}1vge{kRzYsaZE?ofSQ0QV{XPPn0v!d7}IkfY`ElZU^=MUV5981!2)RxSwySnDW4;*{cLN0W%gY3dnNGKEr}MJ@Q8oQzP3UkLD8"
        "xJhYT=RX&dtS~@rI+vRCs4Jzf`L=jMz=rBfQ&R?EF$4cO^rfQyi!A1QJTG7TFHlPZ1QY-O00;m803iUhHXr5_2mk<P8~^|#0001U"
        "a$_%IX=ZO@VQgn*ZEayMbaG*7ZfS03FK};cX=8aVaCx;EO>g5i_8uVrfmJWk)T3!yw1@2YFpEur0n#>TXLb*UAutx5up?0_NhOoX"
        "`oHgcq)3spY^Ok1mssNa{^m=TWmlj3hHtngTiWvGiS!-3J`pNvQZvnD%LP}OZ%Ef`-tsR@k~`TAy&~IA647aP+jXDF?>~Jgvn)G5"
        "J3r%X-$_k2UH{}?TB`4ZGu_F}eJKQ?iU@%RTiLbbmT$WaZSGjx(o#zbti@f~vbK{?RWG|+Mn(cR_;TI-_U@VsM&$yyi7meyB(rdp"
        "!$#WYXEob`Hd{J0+MgicmN!i0#pU@KfoIdTZ70Z*Jll7CvIiy=??iS<GO9Hb8eXcVleDsysva(~MF20U_yoWgKZYM^)8AA0`u%s|"
        "*R5pi3#%SGSt~gFAO|*c+&66I5a=oVp8e5rp`8Rf`^Ra9s*In|S)+%!AbF;q+Lmd_H`xNjo8Fu)@EQFKL<DWwC3(&o2`s;3uYUlf"
        "M=IN)uasszO8>*o1_VG_4oqyinn|QzGyAjHO~m=u$+KKW5(ENFLae}q{7$rB|7?vC9T#OH;Y_iz%+|4Y@p<(MBekp<>+wf!qo7#C"
        "P?^OC>h?!^Y@(`~guFbUU5aPb191BzI8J=Ia#pN50dgdYoK|8|ASM)o>?`pTZB!I@UfyT#)sum-i=$IAJxJjdn=<qPIM-COAVp1z"
        "<@V2lw7{=f{?ZBN!qgdgEL-Ir)49{Mupm>QL9xQnd_EZbi^Mj&AQnL+#mK*DGq5WuJDF!6#V64{3gV=FJ(f<(tqk~w!nt}<kgt|U"
        "N(zBo3ejx7TO^zEM`imJ#3u%hcoqcfrK|4`7lbRKh91i)s}D&Vm8b$y(1=)!QUQZ%6~P@9^FKiM&CT_dw;>UX)rwG}SWhLy0;u7_"
        "qIUcOS@l!T@=y#@ehV)F8o&#>vZrhEmRuYa%NPn_NjoS_pzTl~p}6J`Sll$bV=|N?pNJWziVLNw*u<jQCQvL$4kQ+2+jLYH#e{0x"
        "scDrZS6i@$b_7-<@;U}j4$0dk`PKpyr}Z;PUO~&ww0)<k*N}bg*#_;uB>9x~yQG9inL`-V4al^aCa^2We?yBs2c<ow8RVv=sKAh;"
        "RFnDfV`%8HMlGQE6((c60!hrK_09kXqRmV45)0qhEVGil@f=9qIUjHC(?a3e=K>YdkPHf|mGkka)Vc&WwbRUkPXqIhExO<#hif&2"
        "x?C)hZ&7^?0tzgdL@yGKf^|Md8)*(2X#z0m-+Q*%X77bbX>$FW(b-HFyD|~QWVRJG&E`?pw||A#h%>_vc$vQ9m|%Okj6^vU<Oz32"
        "s2qbnB#2^1hd6!y0YQm%d7km%;QE&p;x!84>bYcp4qP%Uowv{_8W@X5hM`>zJ=1<81tBb;OUT?HK&nnzYkDmT7%QP5pGwVH<+O2j"
        "F^wY0q9l{jL`susUgw{Uizr+hAmTQl0fSs(D#)h!?u$=g>g7Q%9P)Tjfv`BH_1)ty;38ALud&tYr_=iU?#Md1ZH-dIks!Wkq?vJ`"
        "KAx7&-jz<v_D&rVj*<o_2&kl%RwASu<bunX9cYZvU3l4VPK=igeWL{w47UuNc<8nlvsh+0R6!JDD__PlKu~i5XK2i&#m0txk4r3="
        "J+P~VSfsbm)Up<=RB)xm(Pp{VuiqqtNi-iDT_>B~4!o%+BiVNFH<%QBCpimld8_s5iyV&4Q`a2ldc~Nq4al|VuML&D!cnfpWrBiR"
        "cs=040mOsx^PWnGr5krlx?7A_wE)JOW~k9>zz6D?D$^e6X`+zI46)W4Wy&)z5>H~5_~X)5PB1C06Awb1LE{`)90+ypu?xc;6hEC*"
        "cE+W*5Neack5qvNhL%Y);Oa-dVNt##?UL4Y9&t^&Hhe81G^JLLT;F?-1=S5`j<cPOw{-GZ&)vGoGIa9<0Nv#<Q^(1iGf~^qJS)uw"
        "=N*QQ{rS(VNE60$bs=|(=NONF3r`o@Ay1Vy<e3TMqwu$=z{TEIm*c$gkEv++R>h@?tNWOYD<jmpq<s(C<arYD&i#Hrvs%a5@xr{l"
        "+>0W2MNWs#{Yl43idgCKd;)wFehi=H;VDb^=9JmoL-~%ep7GCB+e0^<JY#fnUWrp}oVu5L`deP<X3Y5~7U6l{7#X5xm$q`v92+a("
        "u4H8=)C!!9ZiqBpAH=Ni+@K2^0t-^T4Bf14oZc&Zodv^*vU(wGvi+G5qJtDQ*w}^lI%p^`OFm=mD~xLW)TYA%Z`KxZB|pBq`88c;"
        "j~BB|_rP$fY35RSiqWw}TimHfC;hh4h@X#}y`)KxBv!#icy2Z5HiIi0L#f>Y*NkB9!yaVfm=XM_SF%ZUbmh?IeTmJg$pdeUgvJ<N"
        "?=4PiyvZ4S+oYcr+%)>qd_GKOw@1fZEvl{TzA)UzyA+K!RC@~CTwfhT!By41m4HsApTLQ@DYXMi_5MQzbKiX@vuQuJa%;wQdQC8k"
        "ip(vrvi+4VKB%{CJ$#BIJXlZR<p1Mhr@gABf^V79V?@0?lE(6LG?3y-(s0LS6`kz6_x~1vf6-V&Z${oR^UO5`GZ9^&a&O56K}ODU"
        "SYzNd*(4mR76#ny%tR*yyQ9&r%<xP8@8U)GV%Xv}4)FnNzr~p~PGAs8{e6~;{G$AJL2@J8H{_y#dxc+|Y#mJ#l>nGDF<Zzdj^2CZ"
        "mFE(gM!V_6#ZiEZXi-QS@bk0(0Z>Z=1QY-O00;m803iTufR+o10RRAr0ssIe0001Ua$_%IX=ZO@VQgn*ZEayMbaG*7ZfS03FK}{i"
        "W@&6`Zf7oVd2NzGYurE#h3^Lc2jNo!8@te(4}~0(9!i(8O>+&(SmPZsGjcU@H``xd&8C}@5}$-^z4!D<UDw@x_&7o$EUVB2dz9&b"
        "La+$?5UGbP!~O31D^SspPzgc-6|!mqEP(%dg~}SW@4BvS+QthfcnqE$#+g#W-f{L}hL9E(g`4MvBqYlEeIfVW|HYyA9H(=~t(lt9"
        "mUTxs9dATfs#Duc<j;X0=vDWknDF?Y(FQDr(z8A&KT$3tgsdA-l$qYVyM(JxFa@V$+kibb*6mvTM>tTJ_gV&C3`zJ+GN9A&3pH>q"
        "L!s!ezT6K^xD+d@W1I$~njjjs?2dg6T}5|vn(Rz{V^fd|#dIuE%>t3~tb5~}4R?%2M_FJYd@4|AK{3U6gi7y4lDwyy##U$_6DdjQ"
        ";qJij`P&b0xS$If$P{hMnbnq9)@18bhDU2+jx_bJGrtsqq0qYvxTpuzZJ^uY$u#jqvGSu?PC)9t2>o0`JrQr&jn#_r@4iBw`|h%B"
        "{s2%*0|XQR000O8001EX=*qR9zz6^UFC72?D*ylhb8=%ZVrgb?V_|G(Wo>O?FLZKYX>Mt5XD@PPaBp*IbZ>HbUvp-3E^v9JS>12r"
        "HWGhdVE+eJ_ae1gUmUc1*9CkJcgO<w&|=XIiUK}EU}VZBqOT(<$8M1Sy)*n0DO=8NtA|7)Im4OZkG~nxNs|1+`c87yi4$SRz7vY5"
        "?vb_pcdB^T654J_LshwF+jo6e^YTPKf4nDECkU-;BKeV&U0d<pAhdv7BuVo2&D%GX=o->fwXgXD;f>b(2|j(_1FyQ0);rcTbYXP3"
        "ad*Fcy8Gq7Ki&O&Uwr;__n*%*5=zigNhfq~%w}8gh;ngvq)lJ5OQKt<=%TBtnzyWwm2w)=w{HkMy#NJ*6>Q5DZ+C@YWhb^WH@&j2"
        "xti-!^r2=&1wBmkg0~7fvRM<v7<y}gV02r^ZV)9aME4|TMWZ#c!gUA22PSyM*|xAFn7E_^BF9^>@=_Eka4%@SDsB;BB^?l0VcLtP"
        "+p;<*TUODbR)rHQdeK$9W)iXR_6>A9%z@WFCN{QqQ6L&Q#GaKoVQmR|1?sgg-}=U)<_%Y??{9v5@B3}rKbJq=!fJdl`gnyZ7dxiX"
        "M07PvGV;xeB-+Hpj?wc{E*bfi)&si}q7!Mt4)CD^?^eG+w6tB|Ag)o887&A@c2s1A6Rs2cnQTn2vzu=vj8FqWHnnXF^FOr`IiXts"
        "2ZVsr+AObJ-HU-`8G3^qRzeU~gVAZ`T@5<}=U^S09xY49Hhh2-)F4{Ywe^WmGPD{}WXr3HR6`AG2k6)lwP%F4Qi-9|I2@#Fb#_?O"
        "7yoOMc!uvxFal{85PYoR=*1;#|D1VX1egIcTceEPhDk|x41>RZHc0qDl0h<&<ispitDwnD{<85DQ33O<9K;s1@9R@h?kRwP0q;=7"
        "f|$BVa*y-XM+(lFCEc->8S*U5EP7QD(wa@wp0|(Sj}b9SZGgbp`m2qA2n3!|DLK#<_!zmXS4U9+`=jWQ{*%qiCLfhe{1`hfJ2V}2"
        "(X~AC&4f7G7tT++FU*LG+d}Khf+$_ngZ&EuIN^Poy{?W|UgP=$saczjre+&tk@5Q~k=S82$`9SpZmBpq+Az&vv!84-75T971_Akm"
        "0fXc`eywk`QMcHN<E&!%GH>g5OFdwD^Zu&20W(M8v1Ff&;Xy+KUdzb8VtU58ag6gUBk#x(21R`RrM4IwI>>uf-90%ggkgxwH9_jR"
        "hqjPt`8L_)Ara?6^l}@eY|K@zJ_wDl#P2EysB?U|04d9^sLRMjNui#yt>5NxA_x+g={V0~P9IKcn{RJ!Zv6Pe*yyZaY`a?iBOmjh"
        "D_kJm*NO?;95Z!KKq1reBt=JI%b2EBf|5NgA1h=NL7%K0%Nu;6-EA=oC1fgW64tX7>eOQ{UIEJo@zoSm9F~bLJO$9Yb0E8lllwiF"
        "L<}wP5eWQ$-~#whOe<c90vglsBV>`9?I@HqR{g>O>Tl$uVC;7WNe9wf&37CK3{uG+3m;mFPD)=8>#|3Z2L{G6vJ>6VgJNAcVqh0e"
        "<G-n7SY}DGrw0a{3w0LyhlxwlhLHo8b{bPc^z|Se9xN`U)utf@i}?J4Ff}-+2(UX0Nde7F0KixnEv$fC`$C%xc-Mh|=xTMrU_ZtB"
        "TxJ8)&eWdc%(*W9)5#uioEmJ|*;#<coNEs@aF$M16MFeB-ZSw1!}8Xqdvlb@-AggTe+?vn(bFjdQPw>LI16H-#Ay;49A3gni9_9o"
        "HOJWjW->lxgV~CA`d|foxyMgE({EAWhq+2n*Ntbbrg;1raQ>z0Y_zkP@iDVSUo#Rh<;9CRZ*q)^3#zlY$wC0jCf05-DFg%9v1Cwe"
        "j${IE6NWF7wFFL-Q2Van@F!9|;PNyMX=@{Zn`Y9ZdC>Ic<R$~yE+eiUB?A7(Q_cZrzjQ2(sN3?rP6hZUj}l1mo(Cz!zeVir4_@$6"
        "G!*J@Ce;-d(8muibd3o7*$!Sp&4DLZC-L&HvCZUA6fC?xI%)XpO^`*Nmr$iCX?a#Jf~znzgN*E<*8`~_9hWI&UuH@X8HhPiOFUY`"
        "1Z?Nj6Y6qEzPzcvns791W=RL1pt!m8xW$+s;V|*4j;|n(K<^xZoS&6n<qmwZS1IL-Sp%O}nFAjKxPUZF!z@?e<1y&~;Q(@8eEFf*"
        ">!>?8#GMRZ{2lp*RthT+6t)E<)-|qH3ij>us5Jp>1ApP6GeDp%Z3;ZRV}TchJ~5nTj~5^Yyx+CFgo8(*K$lm=zN@$0pdxWwQ5yRS"
        "6E{#}*)|;Mh}e{KXA%>hz9I<+1{l8<lhegn)fyh96LW%@PHJAFZ~$pThvx2}IocD@KetDcXdA%5zp^coky=eT0Q#PtDaHs--?^XK"
        "OzJycLWM41TGQk@%HGP6^>GozRY&!RQS~T>8gVn(yfF9Si$s@^Xk?o`^7I0ZnJub7h4D1c=UeyDT%$a0&bx5>N1YuapyY0<@eQDk"
        "r+}{2O)c%^zEg$V)7u~3W0Qt|G9iwa^YS#VX2<?>Bw{eaD{O?^*}e88WvtI#T&GdQb8=inN9?032#WBqeO%b<5PdD+SBWdu$eB3m"
        "zGx}qapAq9b@nCZsdW;%yU~n%=CB(aUoeEo%Of!5Uj2A;fLH6`HA>~;XfDdbvd*97?HiC=Ke!#}cUF9b(9mNc73(FQxZSlv-<5l4"
        "u%EOnhi0ht<=pD~-1#Dh2D}wrv$A!+O9vHgczucsFi#|S{~Fxu0)uEjJcYp78`Vu~t3;lfhAF|zWR%g|qc|%ZNZce(y7bkO2^}Tl"
        "vdsPtv6}y>A<fptmyIjrF8x-K+FvY8o3Htr9s>5jAsd}0IgN^P4*)Cy4Ej2?g@Fe)S>$P6W5*s6W@Bzz1NU1v^M~V%!I-l2OT^_="
        "(a9#AL36afeN<ddH81hDu+X0T4iw%b2WXulHbyNzx)L|=4Nr(sRD<>r#QrV3I5u!L|H5nM5ZB?y)XdTQRpk409|Z5?z4RYYO9KQH"
        "000080000X01WZ;&ovML0C+F}044wc0CRF<FJftCZ)0I>XJu_|VJ~!YVQFq@Zf7raWo%__Wpr|RE^v9J8q0Fy#__&Y`wx@ZaxJkT"
        "!H1+>84^|T#v8|#tm8Oq=b)%iLtsdbiB|>yC7Q*m{6Q|M%BSR>Psx|0du9L*9+Z@rhg|{F-P7IEZw#gMJI=yUnnXJyd7hRs;3Q{p"
        "7V&&Y!cwwD#0lT>q=09ZgoKfhXFLgc66{8iSgi{ZGb!Sf#AzamRJxSXmoHwv5OJ2uf~<=;svmyL)1-cgS+TAk3K8>{FP1WmNyrM0"
        "kHjoh&qD?DQ<_wO84!rXqJsNY(K}L1WuTs4z6klUmd?);mgVcT7=(N)0zUr+2s+-&7X*G<axy3TgMLxSoms-4p`7JqAtEGPEoi)3"
        "Dp~ByJoi^(LGOT!hIGG`X~6S*;J7i5Q@Qg+vP|I#K*^ZQ-n==R^wVNnCL%n7HpQ)wMad#x(*sg9<l_X3v;eH9h$*NhA^s5XtRP=2"
        "4SbL?m9}1%O8Emi&Op*}&ZFfxFIZ9L)N%QP$P19&g2&%7l_U}goO1FAvIOh&3;s|5S9unR0#Ule3=_EM0<J9bdx0c>h`;;OpFG+J"
        "{@AgJq!?&gxaGjPo!&WG9ka>vqySao8UJq*Bn*vw%c2tgd*cHmHWR%PBMR<4pdWesMP&mX9TI%;3D5C);x{d=COw~%7G=VtoRc17"
        "8uV|X7zKSQu})~Q`WYEvbM2c%<RnQ8f-b6pkMe0(FB<;^Yu#m<(Vz?&wPamNjzL2i^W5Mf40@rz;)QlYL#FL7Zkb`=X-T!pN}okh"
        "8o*MBqR*RT(YfpOige{k!!9j|N?dMFlB$-UK`G;G^9XC~!T&$lQ$$blc%=D1@QH__DB&NEz}3bzZ$u*t(j1xOMJW?=YR7obGR#uG"
        "N?ilY6hk7@`^ptEwu@30Vx*^-RUl}@nO_zG_3C`%Ch7g4N;xj9Ij+c4@OVgE0|z|x4G=VW0>h0C0JxRorm*>dGVnvllW8cD)tr{a"
        "a&$=@!g7)q5Ef#~RP_t@UJ5Wd%#v<c##ufvd?J~=l-y@|AjJGP415sMF`VSRpRh!~aL6wS{YyfddNCA^iqQi2nA;1HCWD*@X%gm@"
        "X2AK(8}vTqc@X2!3o&1V29_d0CR=FMj`Tl&`v*1o)KHoO^21;<RFR;<Z1TCJ`wDtICTDLa6K^sL4_^KDGxZV|-*%VGD|!?&N+F;J"
        "YsdwIH6$Rg(u!3$HRR}4s^~2-d>RN8o)(gc#7A_BOY*MP3n~GTyCT27&PWikJfG9J1lOWBzboaIk+@8PG6FReJYs8-W?)X@CobW<"
        "c}6lW!R(lOJ<e`k8-*28gxu(BI@CH3xQGT37!+=<eCvEtX-{#B(hTG+MXK};4E@%Y0&-U^C{nofJiX1`Q8Wd9JDT3zPVUGr<{Azk"
        "4NXgA1?p<|8VuTiLkqRRgW(qfL=U$dvI7N-UVhc7C<yyJR6bGXxK-djWKLYy3i+kl2oRe_yLFnUsfqNMu(4n8uOS%6YK<o%5}m|y"
        "XQ+<HJXG~j0fELGoWdAs?1?+MFfgzr;xhKt;cLjQ`le1*SfUkYDR@i{E+Yl+^)*7>9|5M7rofrbFF*<JrW5i;8{4?Cz30xBhv9zu"
        "_5%2vo?QTxmPfOVMy4fFfV5lmAQ4lmpw=~mjo7Ro0wIG)ofGYFx_o`Dmr3w2e-_7-1UrCyF55)T=@}(U5k+(QYPtA)b{5c^>jH`}"
        "e)A<5h}vh5{~kQDSCl;WVtKe8Ynbb?X0US46VijN>GgI<qx3r8uE<vK`}ZlxI3W{q2|;W&nE-QX#OGARD@ve&Vk0Q9IPm#N88B8m"
        "YaT&sRu<&fAIM_YiXIgH-64T{5f*E2I+=V%Z}j%pn1e_-ryoJ3vkCbD_4V6#Zzq*15{T-EFVhVWc=h)D-TCE$jw^JmRta!BqXe{l"
        "Hmf!G`8hjVT*70T6r&tinbYZ%-t5udJ@-uM4RHJE025CVf-`)C&iREVbb0v{p-DdhFs|oP*f(r%*y1Qlqn#@IwSR(18b?*p$2pu("
        "^x5SiTwYc$M(X5XJNNRa`;(Nx#!k{TmffN|1FC!Df51N&V!cT#S^ejqe<$ySEIkuF!QLP@Sq676k}$m|gN@2+p^#9}Y*;=7r&|%o"
        "Oz@C7E<rkfkSy78uVK8<)0MJZqo26s>qL-6xnU4AHW=o4#)MQk&<qZ$emx1f^h1`f7b%nB0DD3{tWq_k4IY4_S1}P$cpjKGqZ=r3"
        "4B-L*%b;{Aq=6ffzm{pi=bvQB8*w)&;{`WRLlAf+;4EgRiXBWRCK#BKNi|6jg>^_7x_eFHu+$}rPW2c=jG39HQ_1K;YckOJ0MdtJ"
        "vdwhZhQRt8p>m2@qe`%FaDmB05NshVz~ZDEQw?+gnpH{kDA<7hp&4j~sy5SZ00b(}fG6q<ZM+Xz8ms}_D9>V*;|i4iEWnZz(8!t7"
        "#QkIpuNU1aBmniIg^Gdv1R@yRNwhz+O;HkACMaOLCbn3QSe8TcV`@$6w8(t;^o8-7oc+k2pj9wm%W{g^Cnf<Bdese2K-VEOBAb4d"
        "C_rBU=Fq1}LlPyOwupWuSt!sZ5?~FS5z-7;To8$@m-Nbj`v~55R$qPlJ*m6MeiH3A83e`dgnT95(=M^9eLP9aZb2U>g&phD^g4a*"
        "Gl)JdakmA3oGf<0Cy7;k@@W#^eescGGR=A3OT->(JxZhC5)53hlT@0sp&nXy$6(@PpgmRXg<@M}9)yKbEYDbSv&X=#svigQ6u!E~"
        "RF<stl8)85q+ec-fl#|7Q}|X*rXpLTD#Ks^`dY!c>w5Iq@Y-RQtvEF4I8Rc}C4N$FY|oWKi15d`l7$Q{Mx6V&tRi<ExH}*USBGtj"
        "Y?D9GS9bGBv5>P|5Tyz`wlRn%YBq?|J~$REVR3I0rqv#*a>;J4Yr4P@)D&tBstKhnct*4X9LKnVtF+sv9JO0x%nD>gsMX}0)KjZP"
        "vBM(Z`}{$Fb+FXeMCj_WiD&7wJ0v=#-yam`x#(9Nfrwh9Yx3`Z=_)3MsK_KE`<@Jkyg)KZpfa*j&d<|+B+*WBd*e-zh|vl*&SAda"
        "LZ!UIiV%!{W13%!p@=i7n1M8;BE<gSxXStl;K6`njcLq8q{nCJbs?c)3bi8woFe|ZF*18SB(GmPC@R2pVU^K50Rz->4P~M6%wSb$"
        "W8j50y&k~TP(am!&`3|byN1tm{8G_0zGfU+LTD=`r#DCUx}^%s$GI}SQ*PCAJ+)qWT_+kfea50?*`gkz)6-oe_I9RRcNeRe?Dwic"
        "S+ZC}J1^gXdg4(jhB-^}5q96ps|5=-E18x_=)IaOr|)L$DoCSLdawA5UoIzCi&R2M?M<@>lBZzz<ki{P`Sjw#EQ~Pkm$^56n>}1*"
        "EX1j$H=AV-hl9n)dIr42P<@ERQ|&AO*S(WBJ40{{>8kDmpbZzg)3Si4ZQAaFg)0sip@6`~cAiME7aA=frEVJXP5BwSWHpx;AQDn*"
        "Niv%>n46xp_ghI3THE2{75*~<?ZTUakAU|wPI3?06wZpl`4FmVJ<FUfrQ=+!SO&c}Y-3b{$|IBJE|`j7mnQ$S&(9XmK9PxaB3-R1"
        "`j|E?QMAe|*<p{X=0RwVM~k#5(%8GS4o(wP$+j{)P2}Bi>tDrKGYmi%y5ICJC%~lx@D0^Is?AwTn}@?m>gN~Frw+7^loWg&fcN|3"
        "xa4Zh@$<DeJqOjW03*oAXh=8e4HAX=xZ1#PT5@$}Kti<)UiC+%)&e=MPVwy#!5l!>2+}Q=%P74c-Ak}CXshqnT=M4GoA}i-uXoF7"
        "I1L*n(=w?`%VZ~LvQO-W#TSO8YiKht(O2;kn1IZ+p543`OQ?IJuXr3Y%vh?SgUzdJ0oP*ys^$+jorwAXEsVhYT7^}jYQuUP3=Fh`"
        "Ru?20juCJwp(C=cAtCiv8#WWL>0qrCtVV#16x<n->`iB_MvI?W!;wW9V$TLLLnk^U!I}pfHMAL$5F{uw^`!}_>`1Ixf&CX-F#!g("
        "W*NTDpuStXHn$x)O@2J4ob1#nNB7ZfeBE?br4o|C<d0g`Tz890sFhBD|FPt92U3>Fri1oZT=UhMkB-qo{XNu0Cos+DnjXZXuzQtD"
        "4|}_iJJ`t_!vP#r-G`Xmd_q;f{|+*HBo${LMIh4!D}r^0v$$Tem>RdCAbyl_8R;VU2&cD(4L;y!Kroh$;p_LNk?)*LKT^maK{6wR"
        "Hja|OZ;TCkmE2L&dx$F>f=Y)XD{#^d^7;!6(mrjr1KyABfRU&}q#iaj;QBQ9UY*X@U*?7g!svR{Zm$w}NCNJQ5XHN_bD9>^^HuQB"
        "k@(j&E7fch-IQ(CLUsiJOq+S+O3zoYtrjr^q9^oQ4;VJafbawCk2r9+XZY!G1yS*e;FxI*L79<zHMgq6&ydvH#Vjv31Ur!uTAFdr"
        "wtTFkk4wIR4qgz@DS%55UmVd?^KZwWkQb0uAOz+m1arB&`C?OutteQI;0q>898C$fO47U#K@RcO0A8ysd9%lc0*iL_1Ws5H9GmQM"
        "9Q7())XCz@YT9eMNC5j@UBe)(x@8vdD8iB-QhvZ)^8MGON`VRADtlweisyPJhxw<nu)MmMrhCC9@qPR#==%e_as<74IwWs~<d})("
        "l;V_m31Mbk4eaJqcQPbBH^)><3v^syaSpUvZ_Ry{xj}8MDsxhI5v)AuS*(CjsZp#4rDY4(n6XD9_~SMdL4lWXZtq%Px<h&jN*S{e"
        "8P+H6q(i2m4uEOBX~UO^9}4WizESt~8jGdytNu){+Ghi`L{y)Ry2arU<lAZ9J8O)&)HhwQ4GrELAE<Yz<2P@*R{^avQWrZEu%+o5"
        "`Da`fY12HOe09}mzwyYn(a$TKs%uAaH~4X$nlq;dr>TqT{V5&T2BU75<;6h3-PWjgO_G{{&Sb2PKI#L6AEa0)6r4V?Uv*;u&romb"
        "JZr1tZ0Y1SjX}Ux1+;;^^@-bp)u=(cmfEes1HL?Qa}H3eOAE>>)WF9hMwj(hhr2^R`a(jyKP9|zna<uVe^Qr?%!U<;eR!7FSE~P6"
        "_GQt-q37|*p1R-<9070Jp-({RpB7O52lnkn*k47$p7X}awJEd`O^q6H?%rCnOye0Kwe_@Nj>-EZ84Khz8MDyE#st;#8mK||E3+2L"
        "p<x)P%~4NqzV465cx?cKpi*+nfb@Ln@rV!G9RYCqsgJw+F}4m~!-kg+ZOdr-cf)?Ii=8bl2MxF}q1Cl=Ea<7DugK{OwT8c*J9gcT"
        "AuYErlQwgZ`EuLhrg{Inp}9D9)3V6Q0;E;N$e?=tTi~=}&S~A-HTS4LsQ<8!hN24Io!Z1okxRPADdGU29oP5qdam!Q;p9WmzM+G^"
        "Oyc+A{s2TjYNcZNR3`QJovZrBf0VAYG1=M&Gl<pQlnUN$+jR9?MrFQMU)6g6KqLPJP)h>@6aWAK2mk;8ApqL}Oy48|0018Z001Na"
        "0047xV=rQ9W^ZF*Y-eR{ZDB8Ta$#w1X>MmPbaG*7Ze?;VaCwDOO>5gg5WO4x9|oNgP-WAm7ePu(VwW6ZLSpDqjCW`4rTd}oM@T|{"
        "d}n3ZF2$ibNUPnqGxOfOsj6x{|JZ`mfkc9*<r-wr&lI3HC}`)6q|m?<5lAoMZ2kD{Zgsc%-dNdGRW+SVCn|Ug5)%r8F;WC=dmqwi"
        "YlQTec1CXw#}E8_yuQ)O3p87@78~K6(%S=W@zPV!mRx%3Nu0w+fl6_qhZ{@6lmo4NdL)#uix7M$nWmGk$C@rU%p1Aub;xu%oeX=0"
        "{mqcOolbxsLl~TP+b&?D7L@Ur;5V$iBe)(udz{rG_-=zq?7FDm7^>Y_;5MMt%%nphvdspQ_O6SG`uOf0i!c8{^xQffIZ_KTg|i4!"
        "iGvgx+t+tdiBRk)bb)?lwu5Y8<Go?X5>2GNf|3+)9PPTjW4fr)#r}Ha_O$=Bs~%j?S#(-aOzqJ18ulG-PlDIrbLw*{QQOfPm+;}s"
        "P~F%Kw%an7#<tAZD{ZI%BAH^`p(fn$UXKASB?_?|-z>_TX^Ee`XZ@N&qC~r$f12ISuK=Bd`}x)Ee*PJ3jtMpdq1c8O$_E19CvCZ@"
        "VOFRBdqA=!8d&dmBuHU4X%v~O21jLf3}(kUBj7>Q23FY^*uAnke^oe24VEx6GdVIV367-q3+5N;thGn^2Q<%eRYUco#4P33;Mil&"
        "oWM&mH9Rbq>-t1#V1E*a%Zb@iGp6LZ_ybT&0|XQR000O8001EXAT5#yr49f9+bRG62LJ#7Oi4pUPE$qQT3d75#ua|gO#Z_TXWEfw"
        "Alh-0v`L<1OODw(l!uh-$>S1OkXr*-WET*{{`#KF-at~aJAJFilUNY;?Adeq&Uem2h%fxP?q#`C;=L|a=hU~~JoD;5m37*5;??|R"
        "A^tAAQQCue_44KSPp{eyL;v%O7yJEwE@`8=vD+67Z{}Wn`wcC<zg_*d7T3!g@#b!Mv$$W}E!X1h-Aa61-xgwZ`(br=^YINnF6gbB"
        "#rl4=c>NKdcoE;t#f_?Urw7o`&HYpOGxH~#3Ad9?BU&XpF~ErjWm_j?SBcVe6|WPuv0`+p5LWfpRAY&ciy#YMuCyDh-i-JQA)Tmj"
        "S5=A4L9A8DQoe%?tQogE@slt${Gj1IQ;x0bhEvsyJ<+W+{lV((ZV+bQDJwuLSgMDEkmIm3R{xuJ41&%r8Fq3Iu<zDNSlMmqC2xy~"
        "4r(hKaZA!q^cXulB^^NtS(2Qf2-pJN4RQ>;;g!_N@$O*P!J4KJ(yH)7Ly8x8B78b_6)Y=F+nO$j@^6a09(F8<x15W&#*!xeXnO<k"
        "lZ+c(AAL3RVrFz2Coc3QuQhvR3vi<aZ=o8x<9`=oFrt*;eS9y7;op%JtY~E?w+g+8JG-&mdDRNB-zl;HgaK_VN%+J%dyRnwVlFi3"
        "LSAw^-J_^l*I?3K*%D=6{P6Nmm$aV&6S4K9q;YTq*d9Fr&a%n{ksx!UIxs}3!J!lJnWnkxe>dYyT)=Almz`ba-iN<vm<L^rD9wsI"
        "PJ#sW2pVgLx`6Jjc8<cILXJ~{ymKb%*05m-ISKjJ9tT}-RjsTA0rcCN4F4H7X-%c!e3G&(dRlj7Gm=3dD@A7p(dbsA0^m?*>S2$O"
        "=4eN7P6bW~$w-od9Dbt+kWlOGXz2xTPopN7a<};xM8&E4vODlI@XOdxy42RR@XK;1JJ3Al3q(lg@KqUti=H(8$683iHX?b&^pHWw"
        "(ODsldW{KYNIienEd&Le10GLKJpna1?gN9nLqVL(t*W#X!=ay^{gbhuPk>?%4@ql+Cycw461odVNeRIQ^k-{j1?ci1bt5-Tz=RC0"
        "3dC8A#!~vQk_jsU>VYlb?Ff5pFL+y%apf>z^&v9_eS?e(ID>kWZ4Z0HO2B=HdR~dIUiZC%y&oaF8neI5O>v{FegJDeD1laSv*TF6"
        "ZO$9QTOf#HLj>hF(xDeSO3n&5!DN6yVhF>%$ZeP!`<*U#IYNNf2B;8_tX4f}@+rnN*vIFMP+(_c!%rZ?`#aCBAPwuFau77+1PR-j"
        "hH?bf>8<Wyn^WJPB6WnXdIAZBI5wfTAV#)#A<6d@#i~~7l(4FomO=(?LnmpKRm}mCqWet7+d$Z0z;v=zm%;fO(8<=aqykdpMKoIW"
        "L@l(CGIj3#H;D6|37qx(aYDo#&bN$a^a&WMZBzxNO<Y6qTzSS4MH#jbErS=HIzW+S$bcomz}`(nbT@7QPCT9jdO~qV%9DaVFK8!%"
        "_!IIGy-oG-*;0}d8iZ@wA47kmz@Rl);~A3kb&NpF;;Cjq7;}OM*RV=8kdD>>NEhgejch2O_7)d*L{Y}hyFg$<Of0XGvCvk7ONm2H"
        "aK*D_D#2O43;a!50ZQryS2sXFL0nePB4cs~H>lQ4pdE_xsIa({R6_kL90gbxnbSm4mRr8aVPz5wnaR<1z=_Mz5f7j(Tf$J!IX@8w"
        "W>rf)2187b8v+wf<9c0=W^|Cst+byJ3~eG&k(()}w;k04h)MJmS@&#6B6`i1V00<+yq(X^CG-&^k4F#b_+@64n-MXyJ=$07K%b3*"
        "zyf-y2;V{Pe4{KQo%&;hKx%N~(ty>uoFhNZaxp;DtGW0U2^jADCK@4dFtHxF^!l(p+XdxGmxHx}8Y(g?3B(l8n+P8<2OyGgQXqr9"
        "8o)d;5&;gIYOj%xbf&wa5O8p8{P7C7k=<g?We&0#4p+5R@RJ4#_+U!J_tR?cJ4oCzbZW2!GOov%J^{ZB=KFC2i@}x<w0$EX>f;ko"
        "o9nQnC!TubJ>R4?hzL#e`ee^D)t<m6_~|d1pFT*$;{OjPUBEik514S!sSJTgfr^ehsY}uGaWf|Z!3SwO`JjjmMwO@+H+7AK5UPV}"
        "0G0W_05Zl7?4O7Xo>qJMNXQcphh|_`g?(k;H`uJ24xCS>Mm+P%m5tP3bbdEKADEJ)<R*@2+CeruC#|Mrtt|jjXoQrGbumYR3wH^9"
        "wCR+umVk*sT4EbO>y8)1GjWIPYcZUF$>3ByzsDZ^5$fPvENTpc)IvExFO2W#nn7<_3AvT{YXUys`(LD5mYBFT&Rvn!@Fb-fk!A7E"
        "5CBp%a_>ex;AtDR<>~-d3+iW_;Rq_913fAhth)2PdM2n$E;@vh1+TQkX+aM2wP{F3S`}JZpEaR1N%`i>VPIZdH85Y%hcRZP3$(6+"
        "ry&ZXi9sB;m{ov=@8@Er@<G#_c5me&q1`bs0XFKuf2I)i3>zfRAisp&MgTqvC}chGf0HWqwD0B`{{*@fsedA?WI@&{#qO?61DzBT"
        "fB<Jdr#gHgFZsA5MBNs3Mise70S9Y1Cjwc{lT%}fe@;#-sd5}QAHQ&!joWUrZFvrqs5kasI0WDsk;QO=mP2<KGu(N)TmuBY7~v=s"
        "+y5=uM6sab&C6_as|HB9A`tCt4nQ3RD0O_oeDAnt3XTGEE>+LMM|**ASt(>#MNT;=VuzG~{(wBnIHS=C*5aSU@)%Zw5K^x)M1Y|L"
        ";Yrb`oP&1A${HW$w3;8AuDV1ti?QI_M)dk@d3V2fb322Cd>qK`n2CNHWa;@<dA?=9a3*z5>`QLTWrfyNf-hx7EozEDb=J6uRT3x3"
        "xg3vngcp1kIz{m{*3IRdH}qNSQrLlzMoEW_*yKvUuSp3E3<|dWIVdWFmdW7BO4Imq&uZ~Y4)l|7n`HWQq9t^l&<AVkHdV;e4U8?$"
        "m|6y2n9a?6M{&kd^>Nw|%YgQgT?P_ub%p0VL=Sa1^Mj5X*@u$Q!rt$=fkRL`W9!WA#4xxQkCQ{_JW}tAJmyFtpJ{|*hshm4EGe?8"
        "@ZT02r93oqF+qE8rLReff?eRi8<S&ED~3~&s#0Awwt?hMA}1gXH;=)oCoo4=4zmz2#o5H7*%vfh%oXf-5{PV^r|(G48Y=Z7#B%9U"
        "fLX%PBrW#{O7Vxu)y65GMv^;W#b-!=HZDAKO~rDu>nT&8(Wl6AsHR@?@I(ib5By?YkwDx#pAaS$POiu%WjgY7990wd#(<hmcw&Qd"
        ")IA*e<p=7L{G9{$x`{Kox%k*Ym38Dh^$2^F8XIVmnO(=ENy*`ubY|1BY%KQVC@gCaaL?nZ5EH{qKHdJWx;oFuNYN~ESO!zwD&Bpv"
        "fTbC5RlF{u%Cj-t&tV#GsWrs<Ldv-j%7jyu;;SdjiQF@OD<&X3dIeL)UTO>dASJ=UXA3oR>QR+B^b;OOqgb_-manpoTb6j`@FO5("
        "U?~pKGUMKrAwV6Nre}9hXnuc@#~df#n8+4qagq1nAWhi^ygBecf?7VF`Q;G6LFpJWsm-<Ok5T*kDlGU8T!ST_Tno&=w0KdD`hcy0"
        "nUruy?~v52;1u{|_8W35xX;&#GqoX@>#uWhqaC$fcyC;bPrz2eIER=KQQOUdJ7DVGu|3Oxf!s-LPnxV1$#EVHT+;Re)y2`nNspVq"
        "j)T#OmoITr2E|}@y%vl0OuW8cFV@k-pBDG;?mpg&PuHu}_40mkyB2q={2PJ0x8i#FyZFaqc>|nC^Bwdf4qjb$7)@AHWh2;>Pc(y-"
        "0lN;+CX>0Sm9l4$7c6?exPO0Jfbo}Ci{;zZV)^UsZ@0_)Li~2Sdh-tSynelSzqtRMV(IPTetEm*uNkg`kPp`@aNNiD*DLYi<LblR"
        "`j)FP-%~btg8@hAVOveN8*~fAP32MCfCH`Rtwugihpr*c@EwZ81jN}amZzA`0dc|;M;O<RAm5o%$4;Bk-e2?5s5igoJ?(}=6#i`v"
        "Pl91_*?TQFx}i(_1(s$3B&i!n6_x^z8k(bmn$WH1W1Dby29Y%6qlZpy8x1V2RF_4(bT1~8;xtG4V(ecqm%y8YMsJAuk$PL4YorU3"
        "u<3xGLpZvnKc6d%sgt^b1D@!)M$^`Qus|-Bt=vv;3UGP&Iw5_h;Ch93{P_hqWEZetzOq1?!xLk?1oTn^aN`&cG{=Fs<(qq~@~M{N"
        "we@kMOeP&8>WutEciyEL*iPmS7tdbI2fgs*jp5MU8dL3cGnq(!hMH~qUgGQ#S;L6B*HSm5<w7N!dh8Mbq5^sD+Z4QPz&Orr!doi`"
        "0fYgGg#LK87lg!ty{sNIUC-A3dlJY7Z;0@zg_j?);itK{F0ojkDFPnj=GUo^X9@CYhg^S}y(gEH&tCBb!dmW(;rS|!P^VY^G|vTs"
        "S1UpofIrevcBSHj^gLkoh<Km~RBeZ!yJRyqHf|Ht7iQD=u@-Tg7YIa1llfi;j)b}6+cF(tH8v&h%pUtE?n0x{$f}v-bVT~jrfc$x"
        "mq^k54Gc|){bNMegsr48v7vNflL37iJ7w|UC(hVv>6%d+^Pkh7>{BQERH}v^hnE14uFeeSrELkC0u_&z%(8p5Hr>$q!89N>H1*i%"
        "@_e>9ol0*Go{eN@I-nJjVI%q9XJMJ~%cxC=<=f>AR^4-7?D;p+ozttAuYQDFu%r1r$c^~3c&%$w%H~(qwlde$@k$>ARmPF0Unds%"
        "Cy3$#Ux=UgUH(Ec0{cXtGAQ}sYT#?S^UVhZdYliW0wz+!XEro7k(1#~07W^Z4)IIhIlc#)wp`y{f!co2*O0NNDvST5iR5JCmD3Mr"
        "aIl=e@FdszABk3ASZuwuQWGV@g)x;aXhY$2-F5_k1iXh-?~Xqo3;lAUX72Q8u+@J6P)h*<6aW+e000O8001EXUAs{>xB>tGHU|Iz"
        "Bme*a0000000000fB^si003ieZf0p`b1!6JbYXL4bYF61Y-M3{WiDlSVQp}1WiD!SZ*EXa0Rj{N6aWAK2mk;8ApoQ_;v)|M006uJ"
        "001EX0000000000004ji`T_s|V{dL|X=igUWMOn+b7gd2a%F5~VRL0JY;R+0E^2dcZcs}B0u%rg000080000X0O_tvAJzc?0Ei6$"
        "02lxO00000000000Du8V1ONbIZ*FF3XLB!QcyMKMX>Db0baO6hb8l`?O928D02BZK00;m803iUI&P^j-1ONby3jhEe0000000000"
        "0001h0aOM60Ap`%W@%@0FLGsYZ*ysMZ*qBGb98lNc`j;mZ*EXa0Rj{N6aWAK2mk;8Apl(|JEmp<008X*000^Q0000000000004ji"
        "><Rz?V{dL|X=igUbaG*7ZZ2hbVQp}1WiD!SZ*EXa0Rj{N6aWAK2mk;8Apqx#7AAWE004;s001cf0000000000004jiiVOe%Wq4t2"
        "aBO9BFLGsYZ*ysMZ*qBGb98lNc`s#lVQgP=Z+CMpYIARHY*0%90u%rg000080000X0OeyG4Z;8b02=`S03!eZ00000000000Du8Z"
        "4gdgUcwudDY-MvVa%FIDb7^#Ma(Q2KbaiBTFH%K8L`_95ZDdeO0Rj{N6aWAK2mk;8Apm<V*EZk)004^t003zK0000000000004ji"
        "PY(b9Wq4t2aBO9BFLGsYZ*ysMZ*qBGb98lNc`tKrVQ_P3Z*+4nF=A$9VP-ZqI5ajnGB9H`W@b2LHZx^2IWjXfH85f^I50IeG&wkB"
        "GB{;6Vl!nnH#Rh3GGR0{H)An3Ib$_6E^2dcZcs}B0u%rg000080000X0KArWn%)2a0E+<t0BHaK00000000000Du9u5C8yWcwudD"
        "Y-MvVa%FIDb7^#Ma(Q2KbaiBTFLQ2TaC2yHbaO8;WnyG8WI1DGF)}k_F*7qYWiw(qHZ?UfWo9%rFfn2@WH2){W@2SxIAJ(BGi7Em"
        "HeoPkIbmjHH#IOfF=I6@YIARHP)h*<6aW+e000O8001EX=dStY-~a#siva)tX#fBK0000000000fB_p4003opVQp}1WpgibWpHnE"
        "X>@OLd0%sMb!2%jb8ca9b7*gLb1ySwG%;c_WH>P}Wo2eHGh<^oVPQ2gWjSOsWHdE5G-fqoGcYt|WHDhhFl941Gcq?ZW@Te#Ic709"
        "H#jviIW{h8b8l`?O928D02BZK00;m803iS_E7hIh0001s0RRAL00000000000001h0fG|%0A+Y#ZE$R5b1!mbaBp*IbZ>HbUvqSI"
        "WO*-hZeeh9Xm50LFElh^WMnfjGBsslG%+$ZIA$?pG%{i~Fgav4Ffn8>Wi&N8W;r-9GB#p1WMN`5I50FYI5;t4VPau1Ic736GA?R!"
        "Z*EXa0Rj{N6aWAK2mk;8ApjPL*g@X_004^t003zK0000000000004ji>J$I~Wq4t2aBO9BFLGsYZ*ysMZ*qBGb98lNc`tKrVQ_P3"
        "Z*+4nH8?gfV`5=2H8DA6HDfg~G%+(ZW@ctJWH4khG-fqoIAbw4G&49iVK_N9Ib<+pVPiBoW@0&IGh{h2WH>ZAE^2dcZcs}B0u%rg"
        "000080000X0L%vnVc-A&0Ez(s0BHaK00000000000Du8d761TccwudDY-MvVa%FIDb7^#Ma(Q2KbaiBTFLQ2TaC2yHbaO8?IAUaF"
        "G+|+8Wic{2Fkv!eVq{`4V`MZqF)%hVW;ruuGB{#mH#RvlHD+UGF*7nYW;Zl3F*jl{Gd5#oV>mf3YIARHP)h*<6aW+e000O8001EX"
        "Gdt}O-~a#siva)tX#fBK0000000000fC0A`003opVQp}1WpgibWpHnEX>@OLd0%sMb!2%jb8ca9b7*gLb1ybFF)%e{IAvusV>vQ1"
        "IW}T5F)}w{WjHW0HZe0dWie(sGh;G0WiT)}WHT~kH90sqWnpGyH)3TpV=y-`WH&Bqb8l`?O928D02BZK00;m803iT%jzm!40001s"
        "0RRAL00000000000001h0Uj9u0A+Y#ZE$R5b1!mbaBp*IbZ>HbUvqSIWO*-hZeeh9Xm50LFJd`1G-P2pF=S(5HDfh1I59afWnnZo"
        "H#B5oV_{}EG%`72H(@b0G&M0WVmC24Wnp75WHB@~HexY1Wn^JuW-e-TZ*EXa0Rj{N6aWAK2mk;8Api}^5s}~k004^t003zK00000"
        "00000004jig&F_=Wq4t2aBO9BFLGsYZ*ysMZ*qBGb98lNc`tKrVQ_P3Z*+4nWn^MGIWae4H8o*mWMeckFfn3eIWc25GBY?ZIXE^o"
        "IA&#HGG<{oH8?O~Wn*JCGc;y5WMW}7F*ag1I5lKsE^2dcZcs}B0u%rg000080000X09Mc2<wF7h02mJd04e|g00000000000DuAP"
        "8vp=hcwudDY-MvVa%FIDb7^#Ma(Q2KbaiBTFLZKYX>MO~Z+CMpYIARHY*0%90u%rg000080000X00g{E!)5>g0BisN02}}S00000"
        "000000Du8^9smGya$_%IX=ZO@VQgn*ZEayMUtei%X>?y-E^v8JO928D02BZK00;m803iVLCc-_F1poky5&!@k00000000000001h"
        "0T&+t0CRF<FJftCZ)0I>XJu_|VJ}}_ZDDC{Utcb8c~DCM0u%rg000080000X09jot%})RT09XJ303rYY00000000000DuAHBme+&"
        "a$_%IX=ZO@VQgn*ZEayMVP|D-bT40DX>MtBUtcb8c~DCM0u%rg000080000X0Alb-By#}(0GR>+03HAU00000000000Du8=B>(_("
        "a$_%IX=ZO@VQgn*ZEayMVP|D-bT4dgZ*VSfc~DCM0u%rg000080000X00oszHAnye07?J=03!eZ00000000000Du7^CjbC*a$_%I"
        "X=ZO@VQgn*ZEayMVRLh3baO9XUukY>bYEXCaCuNm0Rj{N6aWAK2mk;8Apn-6WlZA&003bM001Wd0000000000004jit0w>eb8=%Z"
        "Vrgb?V_|G(Wo>O?FJW_YWpr~dXJu_|VKiT5GGZ=jb8l`?O928D02BZK00;m803iUlta6Cs0ssIf3jhEo00000000000001h0n;h~"
        "0CRF<FJftCZ)0I>XJu_|VJ~5Gb7gdMFK1<KZDBNDWi(<gYIARHP)h*<6aW+e000O8001EXgY6oG9RmOWFA4wvF8}}l0000000000"
        "fC2k00047xV=rQ9W^ZF*Y-eR{ZDB8Ab8}^Mb1!snYh`X}dS!B7ZDDR{W@U49E^2dcZcs}B0u%rg000080000X0Looysg3~v00aX7"
        "03iSX00000000000Du8yF#rH_a$_%IX=ZO@VQgn*ZEayMV`yb#YjAIAZggpGXD)DgP)h*<6aW+e000O8001EXhr*_*d<6gi4GsVR"
        "8UO$Q0000000000fB`Zy0047xV=rQ9W^ZF*Y-eR{ZDB8CZ*FF3XD)DgP)h*<6aW+e000O8001EX$R$*uFarPpdI<mk9RL6T00000"
        "00000fC1(>0047xV=rQ9W^ZF*Y-eR{ZDB8CZ*FvQVPkZ2E^v8JO928D02BZK00;m803iTX1w2xN000280000X00000000000001h"
        "0Z}{v0CRF<FJftCZ)0I>XJu_|VJ~E1bYU-FUukY>bYEXCaCuNm0Rj{N6aWAK2mk;8ApmE;9qPLR005~4001EX0000000000004ji"
        "6FmR`b8=%ZVrgb?V_|G(Wo>O?FJxhKVJ~51VQ_S1a&s<lc~DCM0u%rg000080000X0H)DR!O#H!0BHjN02}}S00000000000Du7v"
        "KmY)9a$_%IX=ZO@VQgn*ZEayMWMOn+FJW_ZE^v8JO928D02BZK00;m803iV5)`me10{{TW2><{g00000000000001h0T)650CRF<"
        "FJftCZ)0I>XJu_|VJ~E1bYU-JZ)|K~bZ>GlaCuNm0Rj{N6aWAK2mk;8Apkx|#|_8^000pZ001BW0000000000004jiV?_V}b8=%Z"
        "Vrgb?V_|G(Wo>O?FJxhKVJ~BEZggdMbS`jtP)h*<6aW+e000O8001EXqNsF$@*4mEu3-QGApigX0000000000fB|Vt0047xV=rQ9"
        "W^ZF*Y-eR{ZDB8DVRT_HV|8+2bZKvHE^v8JO928D02BZK00;m803iT=_*tYj2mk=T8~^|$00000000000001h0h(w40CRF<FJftC"
        "Z)0I>XJu_|VJ~E1bYU-KX<=t>Z*z2MV{<NWc~DCM0u%rg000080000X00@0m7;Y8-0KiB903rYY00000000000Du7!aR2~wa$_%I"
        "X=ZO@VQgn*ZEayMWMOn+FJ^UYY+rD1Z)`4bc~DCM0u%rg000080000X06gx%4%Q9;0826e03-ka00000000000Du9&hX4R`a$_%I"
        "X=ZO@VQgn*ZEayMWMOn+FK=dMY-w&~Uvp-3E^v8JO928D02BZK00;m803iUuN$^2S2LJ%}6951p00000000000001h0oaxR0CRF<"
        "FJftCZ)0I>XJu_|VJ~E1bYU-WVRU0?WpgfYc~DCM0u%rg000080000X0E-GY=i37S06Pl+04D$d00000000000Du8uo&W%Ia$_%I"
        "X=ZO@VQgn*ZEayMWMOn+FK~HuXm4&`adlyAX>@rmaCuNm0Rj{N6aWAK2mk;8App;T!S$LR006>t001EX0000000000004jifuaBa"
        "b8=%ZVrgb?V_|G(Wo>O?FJxhKVJ~uJY-M3{WpgfYc~DCM0u%rg000080000X0MQj5E(sL?0MJVS03-ka00000000000Du8n!2ker"
        "a$_%IX=ZO@VQgn*ZEayMWMOn+FLGsVWn^h#bZKvHE^v8JO928D02BZK00;m803iT=t?Q7%4FCXpEdT%`00000000000001h0j1Ue"
        "0CRF<FJftCZ)0I>XJu_|VJ~E1bYU-YWpHnEX>@OLc`k5yP)h*<6aW+e000O8001EXL-SR-Z5#jq#ApBjDgXcg0000000000fB~oE"
        "0047xV=rQ9W^ZF*Y-eR{ZDB8DVRT_Ha%FIDb7^#Ma(Q24Z*p*Tb1rasP)h*<6aW+e000O8001EX{};W?_Xq$0rW^nOApigX00000"
        "00000fB{?q0RVGyV=rQ9W^ZF*Y-eR{ZDB8DVRT_Hb9ZHPZE199E^v8JO928D02BZK00;m803iU+WD>Kr2LJ%o6#xJw0000000000"
        "0001h0g?&<0CRF<FJftCZ)0I>XJu_|VJ~E1bYU-aZ);_4X?kIFX>V>WaCuNm0Rj{N6aWAK2mk;8ApoTYH=F4I007$o001Wd00000"
        "00000004jiiV^_;b8=%ZVrgb?V_|G(Wo>O?FJ*RNY;|FDX>V>XUtei%X>?y-E^v8JO928D02BZK00;m803iTe#NZua2LJ%{7XSb#"
        "00000000000001h0k{(Z0CRF<FJftCZ)0I>XJu_|VJ~HNVQh6_bZKvHFJ*RNY+qw<WMwXJc~DCM0u%rg000080000X05_!f&KeW|"
        "0M|zV03-ka00000000000Du8s8vy`wa$_%IX=ZO@VQgn*ZEayMWp-g~bzyXAZ*DJVb8R?lE^v8JO928D02BZK00;m803iVU1ZD}Y"
        "0RRB{0ssIe00000000000001h0lO~&0CRF<FJftCZ)0I>XJu_|VJ~HNVQh6_bZKvHFKA(MZe??GE^v8JO928D02BZK00;m803iV8"
        ">g^7s3jhFcB>(^{00000000000001h0j)6s0CRF<FJftCZ)0I>XJu_|VJ~HNVQh6_bZKvHFK%IUX?A5_aCvlSZ*DGdc~DCM0u%rg"
        "000080000X0JanEsMi4i02TxQ04e|g00000000000Du9TJpll7a$_%IX=ZO@VQgn*ZEayMWp-g~bzyXAZ*DJea%E&`V{~b6ZgVbh"
        "c~DCM0u%rg000080000X02JkHTtEN-07C!(044wc00000000000Du9yKLG%9a$_%IX=ZO@VQgn*ZEayMX>Mj^a%FB~WiMY}X>MtB"
        "Utcb8c~DCM0u%rg000080000X04kA3%HRh80OT7004M+e00000000000Du8LKmh=Aa$_%IX=ZO@VQgn*ZEayMX>Mj^a%FB~WiMxC"
        "Ze?;|bZKvHE^v8JO928D02BZK00;m803iT%;)EM10RRBx0ssIb00000000000001h0c1!40CRF<FJftCZ)0I>XJu_|VJ~fOWMyn~"
        "FJE72ZfSI1UoLQYP)h*<6aW+e000O8001EXIFgjua|ZwbJ{JH0ApigX0000000000fC0`)0RVGyV=rQ9W^ZF*Y-eR{ZDB8MZ)9a`"
        "b1!0YX=G<*E^v8JO928D02BZK00;m803iSgX`>3i0RR9)0{{RZ00000000000001h0f12f0CRF<FJftCZ)0I>XJu_|VJ~fOWMyn~"
        "FJob2Xk{*Nc~DCM0u%rg000080000X0LVsj7F7fQ0Hq5604@Lk00000000000Du8~Qvm>Ta$_%IX=ZO@VQgn*ZEayMZEs{{Y;!MT"
        "Xk}w-aBpdDbYFCGVQzC~WpXZXc~DCM0u%rg000080000X0Qo1z8*l>v0D}nt051Rl00000000000Du7=SOEZYa$_%IX=ZO@VQgn*"
        "ZEayMZEs{{Y;!MTZ)9a(XL4b1XkT!0Z)#;@bZ>GlaCuNm0Rj{N6aWAK2mk;8Apq*<Bm2k#001Qi001Qb0000000000004ji+gkwu"
        "b8=%ZVrgb?V_|G(Wo>O?FKusRWo&aVWo=?*WMpY>XLBxac~DCM0u%rg000080000X0JH>t$`S(r00IaA04V?f00000000000DuAJ"
        "UI74ea$_%IX=ZO@VQgn*ZEayMZEs{{Y;!MZZfSIBVQgu7VRUJ4ZZ2?nP)h*<6aW+e000O8001EXzgbmT)e-;zD@p(WD*ylh00000"
        "00000fB`>Z0RVGyV=rQ9W^ZF*Y-eR{ZDB8MZ)9a`b1!XWZEtdUUu9-yX=7<+Zgehic~DCM0u%rg000080000X0LG+}9mfU$0E-p?"
        "04o3h00000000000Du8qbpZf#a$_%IX=ZO@VQgn*ZEayMZEs{{Y;!MdWpZa_WM5%)bY*UIX>V>WaCuNm0Rj{N6aWAK2mk;8Apjne"
        "qI9tW004sq001BW0000000000004jiZF~U$b8=%ZVrgb?V_|G(Wo>O?FKusRWo&aVZEs{{Y%XwlP)h*<6aW+e000O8001EXMVuYy"
        "<_`b>pE&>kApigX0000000000fB{^80RVGyV=rQ9W^ZF*Y-eR{ZDB8MZ)9a`b1!aTbZK^FE^v8JO928D02BZK00;m803iUYS>d*S"
        "2LJ#~7XSby00000000000001h0fCPJ0CRF<FJftCZ)0I>XJu_|VJ~fOWMyn~FK}{YbaG*7Ze?UHaCuNm0Rj{N6aWAK2mk;8ApjLe"
        "czT-!001Nq001Na0000000000004jiLze*nb8=%ZVrgb?V_|G(Wo>O?FKusRWo&aVa%E;^a%FB~WiD`eP)h*<6aW+e000O8001EX"
        "gBY%lJq7>(yAS{XA^-pY0000000000fB_(#0RVGyV=rQ9W^ZF*Y-eR{ZDB8MZ)9a`b1!mlb#!TNXD)DgP)h*<6aW+e000O8001EX"
        "#sX1W-V^`;NL2s;Bme*a0000000000fB~JO0RVGyV=rQ9W^ZF*Y-eR{ZDB8RWpHnEX>@OLd0%sMb!2%iaCuNm0Rj{N6aWAK2mk;8"
        "Apo`dk8j=u007w#000~S0000000000004jiy|@7Yb8=%ZVrgb?V_|G(Wo>O?FLPsIW@c|}WG--dP)h*<6aW+e000O8001EX+fWH2"
        "jsO4vy#N3JB>(^b0000000000fC1IM0RVGyV=rQ9W^ZF*Y-eR{ZDB8Ta$#w1X>MmPUtei%X>?y-E^v8JO928D02BZK00;m803iU$"
        "NV(en1^@s6761S!00000000000001h0j9qJ0CRF<FJftCZ)0I>XJu_|VJ~!YVQFq@Zf7rJb#h^JWn^D-W^^uac~DCM0u%rg00008"
        "0000X0EMl(<sJb50ObJy04V?f00000000000DuAN#sL6xa$_%IX=ZO@VQgn*ZEayMbaG*7ZfS03FJx(RbZKmCVRUJ4ZZ2?nP)h*<"
        "6aW+e000O8001EXY;1#b7!Cjcvo8PuDF6Tf0000000000fB{d)0RVGyV=rQ9W^ZF*Y-eR{ZDB8Ta$#w1X>MmPZDnn5a(Q2Ha&KZ~"
        "E^v8JO928D02BZK00;m803iT(E(*P%0{{TV2mk;l00000000000001h0j|~o0CRF<FJftCZ)0I>XJu_|VJ~!YVQFq@Zf7rVVrpe$"
        "bZK^Fb1rasP)h*<6aW+e000O8001EXuI%r2xEKHcoNfRBC;$Ke0000000000fB}=*0RVGyV=rQ9W^ZF*Y-eR{ZDB8Ta$#w1X>MmP"
        "Z)Rp}X>Mg-b7pidaCuNm0Rj{N6aWAK2mk;8Apo>CALbJX003qj001KZ0000000000004jik@Eonb8=%ZVrgb?V_|G(Wo>O?FLZKY"
        "X>Mt5XD@JXY-wY8E^v8JO928D02BZK00;m803iTufR+o10RRAr0ssIe00000000000001h0p<Gv0CRF<FJftCZ)0I>XJu_|VJ~!Y"
        "VQFq@Zf7rWa&Km7Y-w(1E^v8JO928D02BZK00;m803iVA%C(=s2mk;t9RL6;00000000000001h0kZu80CRF<FJftCZ)0I>XJu_|"
        "VJ~!YVQFq@Zf7rYWpHnEX>@OLd0%s8bS`jtP)h*<6aW+e000O8001EX4Ds~OH4p#*crX9}CIA2c0000000000fC0M(0swPzV=rQ9"
        "W^ZF*Y-eR{ZDB8Ta$#w1X>MmPbY*O1ZDn+Fc`k5yP)h*<6aW+e000O8001EX+W}19Bmw{c9|QmZBme*a0000000000fB`fY0swPz"
        "V=rQ9W^ZF*Y-eR{ZDB8Ta$#w1X>MmPbaG*7Ze?;VaCuNm0Rj{N6aWAK2mk;8ApjsPk_V*@007%6000L70000000000004jini&EB"
        "Oi4pUPE$otO9ci10000{08s!#8~^}hC;|Wg00"
    )
    bundle_sha = "c1da4ac9fcc8b8bee40e4f7b1355cd188bdfc79cde4a3e8474a54e1d1b542b49"
    bundle_bytes = base64.b85decode(bundle_payload)
    if hashlib.sha256(bundle_bytes).hexdigest() != bundle_sha:
        raise ValueError("Bundled source checksum mismatch.")
    runtime_root = (Path("outputs/repository_runtime") / bundle_sha[:16]).resolve()
    runtime_root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(bundle_bytes)) as _archive:
        for _entry in _archive.infolist():
            _destination = (runtime_root / _entry.filename).resolve()
            if not _destination.is_relative_to(runtime_root):
                raise ValueError("Bundled source path escape.")
            _destination.parent.mkdir(parents=True, exist_ok=True)
            _destination.write_bytes(_archive.read(_entry))
    if "bifocalgemma" in sys.modules:
        _loaded_root = Path(sys.modules["bifocalgemma"].__file__).resolve().parent
        if not _loaded_root.is_relative_to(runtime_root):
            for _module_name, _module in list(sys.modules.items()):
                if _module_name == "bifocalgemma" or _module_name.startswith("bifocalgemma."):
                    _file = getattr(_module, "__file__", None)
                    if _file and Path(_file).suffix == ".py":
                        _path = Path(_file).resolve()
                        if not _path.is_relative_to(_loaded_root):
                            raise RuntimeError(
                                "Restart runtime: mixed BifocalGemma source modules."
                            )
                        _expected = (
                            runtime_root / "src/bifocalgemma" / _path.relative_to(_loaded_root)
                        )
                        if not _expected.exists() or _expected.read_bytes() != _path.read_bytes():
                            raise RuntimeError(
                                "Restart runtime to switch BifocalGemma source versions."
                            )
    sys.path.insert(0, str(runtime_root / "src"))
    from bifocalgemma.config import get_experiment
    from bifocalgemma.data.repository import SOURCE_REVISIONS
    from bifocalgemma.models.v2.native import build_native_model
    from bifocalgemma.models.v2.pretrained import load_local_native_model
    from bifocalgemma.repository_study import (
        fetch_tokenizer_assets,
        generate_run,
        prepare_run,
        read_source_rows,
        score_run,
        write_json,
    )
    from bifocalgemma.training.memory_probe import tensor_bytes

    environment = {
        "torch": str(torch.__version__),
        "cuda": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "source_sha256": bundle_sha,
    }
    mo.vstack([mo.md("## 2 · Runtime/source"), mo.ui.table([environment], selection=None)])
    return (
        Path,
        SimpleNamespace,
        torch,
        HfApi,
        hf_hub_download,
        os,
        sys,
        json,
        runtime_root,
        get_experiment,
        fetch_tokenizer_assets,
        prepare_run,
        generate_run,
        score_run,
        read_source_rows,
        write_json,
        build_native_model,
        load_local_native_model,
        tensor_bytes,
        SOURCE_REVISIONS,
    )


@app.cell
def _(mo):
    hf_form = mo.ui.text(kind="password", label="HF read token (optional)").form(
        submit_button_label="Apply HF token"
    )
    hf_form
    return (hf_form,)


@app.cell
def _(HfApi, hf_form, mo, os):
    hf_token = hf_form.value or os.environ.get("HF_TOKEN") or False
    auth_status = "anonymous"
    if hf_token:
        HfApi(token=hf_token).whoami()
        auth_status = "verified read/download credential"
    mo.md("## 4 · HF authentication\n" + auth_status)
    return (hf_token,)


@app.cell
def _(
    DATASET_REPO_ID,
    DATASET_REVISION,
    DATASET_COLLECTION,
    RUN_TRAINING,
    CURATED_POOL_DIR,
    CORPUS_KIND,
    TOKENIZER_CACHE,
    CONTEXT_LIMIT,
    TRAIN_DECODER_LIMIT,
    Path,
    get_experiment,
    fetch_tokenizer_assets,
    hf_token,
    mo,
):
    from bifocalgemma.data.offline_sft import load_curated_pool, validate_curated_pool_limits
    from bifocalgemma.data.releases import fetch_collection
    from bifocalgemma.training.curated_sft import fit_curated_sft, prepare_curated_training

    full_pool_status = {"status": "DISABLED"}
    if RUN_TRAINING:
        fetch_collection(
            DATASET_REPO_ID, DATASET_REVISION, DATASET_COLLECTION, CURATED_POOL_DIR, token=hf_token
        )
        _rows, _manifest = load_curated_pool(CURATED_POOL_DIR)
        _kind = "repository" if "snapshot_policy" in _manifest["report"] else "instruction"
        if _kind != CORPUS_KIND:
            raise ValueError("Selected HF collection has another corpus kind.")
        _limits = validate_curated_pool_limits(
            _manifest, encoder_limit=CONTEXT_LIMIT, decoder_limit=TRAIN_DECODER_LIMIT
        )
        for _name in ("BifocalGemma-E2B-E2B", "BifocalGemma-E4B-E4B"):
            fetch_tokenizer_assets(get_experiment(_name), Path(TOKENIZER_CACHE), token=hf_token)
        full_pool_status = {
            "status": "READY",
            "syntax_diagnostics": _manifest["report"].get(
                "syntax_diagnostics_policy", "not_collected_in_this_frozen_pool"
            ),
            "limits": _limits,
            "train": len(_rows["train"]),
            "dev": len(_rows["dev"]),
            "fingerprint": _manifest["fingerprint"],
        }
    mo.vstack(
        [
            mo.md("## Frozen HF corpus (all six configurations)"),
            mo.ui.table([full_pool_status], selection=None),
        ]
    )
    return (full_pool_status, prepare_curated_training, fit_curated_sft)


@app.cell
def _(
    SOURCE,
    ROWS_FILE,
    PREPARED_DIR,
    TOKENIZER_CACHE,
    RUN_PREPARE,
    FETCH_SOURCE_ROWS,
    FETCH_TOKENIZERS,
    ALLOW_GIT_NETWORK,
    CONTEXT_LIMIT,
    RAW_LIMIT,
    TASK_LIMIT,
    PER_REPO_LIMIT,
    SOURCE_REVISIONS,
    Path,
    SimpleNamespace,
    runtime_root,
    get_experiment,
    fetch_tokenizer_assets,
    hf_hub_download,
    hf_token,
    prepare_run,
    json,
    mo,
):
    preparation_result = {"status": "DISABLED"}
    if RUN_PREPARE:
        _spec = get_experiment("BifocalGemma-E2B-E2B")
        if FETCH_TOKENIZERS:
            fetch_tokenizer_assets(_spec, Path(TOKENIZER_CACHE), token=hf_token)
            fetch_tokenizer_assets(
                get_experiment("BifocalGemma-E4B-E4B"), Path(TOKENIZER_CACHE), token=hf_token
            )
        if not SOURCE.startswith("fixture") and not ROWS_FILE and not FETCH_SOURCE_ROWS:
            raise ValueError(
                "Real sources require ROWS_FILE or explicit pinned fetch; fixtures are never relabeled as real data."
            )
        _snapshots = (
            runtime_root / "examples/repository_study/snapshots"
            if SOURCE.startswith("fixture")
            else None
        )
        _rows = (
            Path(ROWS_FILE)
            if ROWS_FILE
            else runtime_root
            / "examples/repository_study"
            / ("eval_rows.jsonl" if SOURCE == "fixture_eval" else "train_rows.jsonl")
        )
        if FETCH_SOURCE_ROWS:
            _files = {
                "swe_fixer": "swe-fixer-train-110k.jsonl",
                "swe_rebench_v2": "data/train-00000-of-00001.parquet",
                "resq": "dataset.json",
            }
            if SOURCE not in _files:
                raise ValueError("Source-row fetching is only for the three pinned real sources.")
            _ref = SOURCE_REVISIONS[SOURCE]
            _rows = Path(
                hf_hub_download(
                    _ref[0], _files[SOURCE], repo_type="dataset", revision=_ref[1], token=hf_token
                )
            )
        if not _rows.is_file():
            raise FileNotFoundError("Select source rows or enable explicit pinned source fetch.")
        _args = SimpleNamespace(
            source=SOURCE,
            rows=_rows,
            snapshots=_snapshots,
            git_cache=Path("outputs/git_snapshots"),
            tokenizer_cache=Path(TOKENIZER_CACHE),
            fetch_tokenizers=False,
            allow_network=ALLOW_GIT_NETWORK,
            study_config=runtime_root / "configs/repository_study.json",
            output=Path(PREPARED_DIR),
            context_limit=CONTEXT_LIMIT,
            raw_limit=RAW_LIMIT,
            limit=TASK_LIMIT,
            per_repo_limit=PER_REPO_LIMIT,
        )
        prepare_run(_args)
        preparation_result = json.loads(
            (Path(PREPARED_DIR) / "preparation_report.json").read_text(encoding="utf-8")
        )
    mo.vstack([mo.md("## 5 · Preparation"), mo.ui.table([preparation_result], selection=None)])
    return (preparation_result,)


@app.cell
def _(
    RUN_TRAINING,
    CURATED_POOL_DIR,
    full_pool_status,
    prepare_curated_training,
    EXPERIMENT,
    TOKENIZER_CACHE,
    get_experiment,
    mo,
):
    train_data = dev_data = training_manifest = None
    if RUN_TRAINING:
        if full_pool_status["status"] != "READY":
            raise ValueError("Load the pinned HF dataset first.")
        train_data, dev_data, training_manifest = prepare_curated_training(
            get_experiment(EXPERIMENT), CURATED_POOL_DIR, TOKENIZER_CACHE
        )
    mo.md(
        "## 6 · Matched cohort/tokenization\n"
        + ("All frozen train/dev IDs prepared." if training_manifest else "Training disabled.")
    )
    return (train_data, dev_data, training_manifest)


@app.cell
def _(
    RUN_TRAINING,
    RUN_GENERATE,
    training_manifest,
    EXPERIMENT,
    MODEL_CACHE,
    HEADROOM_GIB,
    torch,
    Path,
    get_experiment,
    build_native_model,
    tensor_bytes,
    hf_hub_download,
    hf_token,
    mo,
):
    model_paths = None
    model_guard = "Actions disabled; no weights downloaded."
    if RUN_TRAINING or RUN_GENERATE:
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            model_guard = "SKIPPED: supported BF16 CUDA is required."
        else:
            _spec = get_experiment(EXPERIMENT)
            _blueprint = build_native_model(_spec, tiny=False, device="meta", dtype=torch.bfloat16)
            _weights = tensor_bytes(_blueprint.parameters())
            del _blueprint
            _free, _total = torch.cuda.mem_get_info()
            if _free < _weights + int(HEADROOM_GIB * 2**30):
                model_guard = "SKIPPED_HEADROOM: free VRAM below weights + configured reserve."
            else:
                _dec = Path(
                    hf_hub_download(
                        _spec.decoder.model_id,
                        "model.safetensors",
                        revision=_spec.decoder.revision,
                        cache_dir=MODEL_CACHE,
                        token=hf_token,
                    )
                )
                _enc = (
                    Path(
                        hf_hub_download(
                            _spec.encoder.model_id,
                            "model.safetensors",
                            revision=_spec.encoder.revision,
                            cache_dir=MODEL_CACHE,
                            token=hf_token,
                        )
                    )
                    if _spec.encoder
                    else None
                )
                model_paths = (_dec, _enc)
                model_guard = (
                    "Pinned sources cached. Guard is not a guarantee of full-shape training."
                )
    mo.md("## 7 · GPU preflight/checkpoint cache\n" + model_guard)
    return (model_paths, model_guard)


@app.cell
def _(
    RUN_TRAINING,
    RESUME_FROM,
    fit_curated_sft,
    RUN_DIR,
    EXPERIMENT,
    MAX_STEPS,
    EPOCHS,
    ACCUMULATION,
    model_paths,
    model_guard,
    training_manifest,
    train_data,
    dev_data,
    Path,
    get_experiment,
    load_local_native_model,
    write_json,
    torch,
    mo,
):
    from bifocalgemma.training.telemetry import TrainingDashboard

    _dashboard = TrainingDashboard(
        mo, experiment=EXPERIMENT, device="cuda" if torch.cuda.is_available() else None
    )
    training_result = {"status": "DISABLED", "reason": model_guard}
    if RUN_TRAINING and model_paths:
        if not RESUME_FROM and Path(RUN_DIR).exists() and any(Path(RUN_DIR).iterdir()):
            raise FileExistsError("Choose a fresh RUN_DIR or an explicit RESUME_FROM checkpoint.")
        if training_manifest is None:
            raise ValueError("Prepare matched train/dev data first.")
        _model, _provenance = load_local_native_model(
            get_experiment(EXPERIMENT), model_paths[0], model_paths[1]
        )
        try:
            _dashboard({"stage": "model_loaded"})
            training_result = fit_curated_sft(
                _model,
                train_data,
                dev_data,
                Path(RUN_DIR),
                training_manifest,
                max_steps=MAX_STEPS or None,
                epochs=EPOCHS,
                accumulation=ACCUMULATION,
                resume_from=RESUME_FROM or None,
                event_callback=_dashboard,
            )
            write_json(Path(RUN_DIR) / "data_manifest.json", training_manifest)
            write_json(Path(RUN_DIR) / "training_report.json", training_result)
            write_json(Path(RUN_DIR) / "model_provenance.json", _provenance)
        finally:
            del _model
            torch.cuda.empty_cache()
    mo.vstack([mo.md("## 8 · SFT (one model)"), mo.ui.table([training_result], selection=None)])
    return (training_result,)


@app.cell
def _(
    RUN_GENERATE,
    EXPERIMENT,
    PREPARED_DIR,
    PREDICTIONS_FILE,
    ADAPTER_CHECKPOINT,
    TOKENIZER_CACHE,
    CONTEXT_LIMIT,
    GENERATION_DECODER_LIMIT,
    MAX_NEW_TOKENS,
    TASK_LIMIT,
    model_paths,
    model_guard,
    Path,
    SimpleNamespace,
    generate_run,
    mo,
):
    generation_result = {"status": "DISABLED", "reason": model_guard}
    if RUN_GENERATE and model_paths:
        _args = SimpleNamespace(
            prepared=Path(PREPARED_DIR),
            experiment=EXPERIMENT,
            decoder_checkpoint=model_paths[0],
            encoder_checkpoint=model_paths[1],
            adapter_checkpoint=Path(ADAPTER_CHECKPOINT) if ADAPTER_CHECKPOINT else None,
            tokenizer_cache=Path(TOKENIZER_CACHE),
            context_limit=CONTEXT_LIMIT,
            decoder_limit=GENERATION_DECODER_LIMIT,
            max_new_tokens=MAX_NEW_TOKENS,
            limit=TASK_LIMIT,
            output=Path(PREDICTIONS_FILE),
        )
        generate_run(_args)
        generation_result = {
            "status": "COMPLETED",
            "predictions": PREDICTIONS_FILE,
            "backend": "eager/no temporal cache",
        }
    mo.vstack([mo.md("## 9 · Generation"), mo.ui.table([generation_result], selection=None)])
    return (generation_result,)


@app.cell
def _(
    RUN_SCORE,
    PREPARED_DIR,
    PREDICTIONS_FILE,
    SCORE_FILE,
    PYTHON_EVAL,
    VALIDATE_GOLD,
    ALLOW_CODE_EXECUTION,
    Path,
    SimpleNamespace,
    sys,
    score_run,
    json,
    generation_result,
    mo,
):
    score_result = {"status": "DISABLED"}
    if RUN_SCORE:
        _args = SimpleNamespace(
            prepared=Path(PREPARED_DIR),
            predictions=None if VALIDATE_GOLD else Path(PREDICTIONS_FILE),
            output=Path(SCORE_FILE),
            python_eval=Path(PYTHON_EVAL or sys.executable),
            allow_code_execution=ALLOW_CODE_EXECUTION,
            validate_gold=VALIDATE_GOLD,
            timeout=30,
        )
        score_run(_args)
        score_result = json.loads(Path(SCORE_FILE).read_text(encoding="utf-8"))
    mo.vstack(
        [
            mo.md("## 10 · Native CPU score / gold validation"),
            mo.ui.table(
                [{key: value for key, value in score_result.items() if key != "results"}],
                selection=None,
            ),
        ]
    )
    return (score_result,)


@app.cell
def _(preparation_result, training_result, generation_result, score_result, mo):
    mo.vstack(
        [
            mo.md(
                "## 11 · Run summary\nPreparation validates static patches; gold validation is not a model benchmark score. Empty/missing predictions and infrastructure errors remain visible."
            ),
            mo.ui.table(
                [
                    {
                        "stage": name,
                        "status": result.get("status", result.get("purpose", "prepared")),
                    }
                    for name, result in (
                        ("prepare", preparation_result),
                        ("train", training_result),
                        ("generate", generation_result),
                        ("score", score_result),
                    )
                ],
                selection=None,
            ),
        ]
    )
    return


@app.cell
def _(SOURCE_REVISIONS):
    # This cell defines a pytest test; notebook actions remain disabled by default.
    def test_actions_disabled_by_default():
        assert SOURCE_REVISIONS["resq"][2] == "eval"
        assert SOURCE_REVISIONS["fixture"][2] == "train"

    return (test_actions_disabled_by_default,)


if __name__ == "__main__":
    app.run()
