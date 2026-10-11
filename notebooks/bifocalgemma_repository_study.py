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
    RUN_DIR = "outputs/repository_study/training_enc8k_dec2k_v2"
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
        "4GY_?{{T=+0|XQR000O8001EXBIo{RF9QGoa|r+d9RL6Tb8=%ZVrgb?V_|G(Wo>O?FJo_RbaG*1baO6nd7V~mYuhjo{vODG5dP#q"
        ">$Urs(b0}(P}thAhA~)&QS5Ufq8Cp(cWv2^-$}L;yGhe@h7fc*J$Lt<?y;jNx;!}@GocY|!834d4)_N^v&?9Rw#rZ#&F0c%E0?fa"
        "0G6A~y}P`aU8K{i>Ez4R^(0Yw5=GH)FdP)tD3){NndHt9oKU5)xG^2ELV(O0&ywR(ih1ibtt?E96Ax3ed$`7q#%5tX9GtI_VnTQV"
        "eKxZdjD~|?W8a{p^W^{dSx6WU2aFycWz49WdP$WbXlyLCEr)`oDV@bsXn|=OJCMbw0cb(kSwF?#l=z*|#!(c(HxVx7b7y3Q5RaI#"
        "EJ~7yn67tAZqWtn-CeM*p@MPTw{`OHkLAJv`_5$r6Kjo)qfOlb-Z51<q+T)$lB2i>HodvM`q69(?L-a?m%yCnrCS<IqR|@(zHLr6"
        "xmnuikUZxO(#$9&uwyN#DQ?WLDkc2v_v(lxN%HH^9JsKP2d>I>kEtZ&I)hZ?op{bvHiNFAl)Z%7^2a+0Yk-}rx(LYa0x%^av-(o8"
        "zwFv*pu6G!4k}<~R-nZb<hv@NNt#+H!~=z9Tw37yBfFX%7se`%2>FL5hj5WO6&@OyNlITY+CB6k+Npe*Akdk~!IE*^7Se*Y7xiS+"
        "9%aaBk7Gm3%MJ>VdtSyY7a0;?K?!0SJzhynNq#*0XMS_*F1F0w!teL=`xMxJR)Q|8HoVu1WCq%qXLWi4wU@M1aktv#+19%ku624H"
        "PVza(EsTxmpa~*!M2maTTvfVJwR>$FvEz>{>Q)|g^uPjER`-aB`)UzXWQ(PQA~*O)d;I$qzH(@by952-6r^T!x6`Pl_Pn&F1dERw"
        "rrV0bnEP0iU#;?<-)?`l*^T#cB8^_eqa>@8t5ke^N_gy)){>?=RYJ!n`_fv^*MFz=0*eQxwQI`05}v%Nkh<6em1ZGu=O~cuJhL}~"
        "_y#E_z4KE~sqa;D<<m+vS6w~c2=PGyC-?N$<UW)aeB0JHr`|3Vx2q1at#m)W1$v#|Lms}DYLn+&mDsZhU07a!JJH7i>6(!A?;4MP"
        "P)h>@6aWAK2mk;8AplhcJW_)I006Q8001EX0047xV=rQ9W^ZF*Y-eR{ZDB8DVRT_HUtei%X>?y-E^v8OjX@5=Fbo9mmHLO3AJ7jV"
        "PW*r?LXk;ah~&hT;|ss1MO?V<>}ZJSDw~ZO=5G`ZMz$W#IR>~FaN#Y)#|bX1-^Y-^Dv@8cKT!v|1!~(=n)Hye5Yar%)9TjnsH<hx"
        "r?ke)Aurifd9NLcd0@|T^~-NP#FSVyC7f}ie@gdx`T$T%0|XQR000O8001EXXTKfly8-|JsRaN4ApigXb8=%ZVrgb?V_|G(Wo>O?"
        "FJxhKVJ~51VQ_S1a&s<ld5u&}Ya1~Ty$A9ihV8`xdpC#BgG&phc8VcQO%vKfDPb&)ydqjjNZL52^uKqsUVnv9B4gI_&c~bgo>UaY"
        "^zv#9DA6Yh5dsNkV+yTE-~)}6euik#$J(V>!W|JvCt@|7+)o}B^Tq6Demg50Sr$cces+GQf@?rHV@M>;l}cO+(2aK?!JIo_a-om%"
        "m};Y!ySEG8?kg;{asthYng&ZolmdzrccI_OqYkrYY`me@GNM4GoRPbPYr@n9x^?M0q1?<u$SCqh7>p67m@rbBY*`qep8>Dv+8_u%"
        "LeW|d(kn}{n=!wXq(F1*18q^nK_sJw(4f_dVj6ez3?X`>dBcoe!06MSf?p>i{O&vE#XMfO8SEImw%l@ugR=!%XEjfG%D@oTI@REw"
        "C>Zp62;~lIUQ-9wB}jppV11fKCuk@}Tu}s!z`0s!?zU`K5nO;JSy49_9wA{^QOZ$=a;%58+O9xb-)3V>L~0qAq9khmzdG^}+n`d~"
        "@Z?=W?0~FwAX%P_eb{jENUw@EI0SWBY~ErBs+eaU8nOxgWF5*aXMZ7zni^zEywA<i%DtQ66k>VQ|KivicRy!$%vRl-V{qz~i(a~S"
        "aFgu^lIPlvTw|=^35}MR_j&N?<$ER!;+kTvSo+O}J!1@~VNP?<Rr%6@jA7IXQsrM=a&e6Db0_Em#pCt;|8X8_F7-+{N!8X2o^w?y"
        "UJU+a)4a7@2FBr4|2(NQlWzmZCite5(UyP@??XQKENQItL8g2RmZThCP05W=ady5)L=DlfrUPd0VsHH3^evmIuT@#8mCA<S8g~q-"
        "rcFLr{0(7~x9k#DD3oyPfSyUTx$kgWG0+-=%;)3B*Q}^VzFnN+G~Te#xVuS9SOh0~%scyWOibFCBaQoO^OQ^5`e3R9YaH?qP)h>@"
        "6aWAK2mk;8ApoY)O~KFs003zN000~S0047xV=rQ9W^ZF*Y-eR{ZDB8DVRT_HVRLjYaCvo8%Z}SH47?lgKM+3c0*SYmUQ)DZi+x}("
        "X_4esECNQNZMCu_NOZB;uP+rp+Cyp^hD;4-$Pryt)&0Bo8)yi5hy?V?5fvNnz$EY?uh7Ea*jPa5kyS?;@MLiYX=;okT~}3A7G=>y"
        "--AX*Wicg6VEW)A{|zmmlWDv{+fv_St-QM^FZdk=pA7qW-a8Db^L#BWDQp^V^_=xM_ObQ6Bh=@I7=6rwW$}3_;6`ZP$bCG-fo{v<"
        "5`%;PMefV~FcG9pS%6&5YdZ2dNC0>UmVOw=E2!)GWdvvn++Ewuw^%bfdgLr{#KgzR>vS_Rq^Z5lm6liFqiKyC!IbZtCJ`@s|ES9A"
        "W!Aoh^{4-JB**kP<iG|Fk!US~Oc73JCWz=-IWb9|W#J-g-UOjK%#FxuDx%IUX38B2p9Ak`nWG>ua!TBbBZbtb?4XTn#gSl*G7KF$"
        "ZF9so@b%mNE}*VQ#)nsvIPXaPPZ2}+J^G)d#2S!6NkcDHKHsMgAS^P1&NHB!trn0xZBB!y?Y3U}9v2Z!qN{^49moYOe?_%|>bVz#"
        "dvXj`ycw{iyWbR(jQ%<pCu0u*<x?t^UM6Dzt!Z6OswYXAvOiEu0|XQR000O8001EX<JN{j4FdoG#|Z!cApigXb8=%ZVrgb?V_|G("
        "Wo>O?FJxhKVJ~BEY;0k4Z*nehd6iYmZrd;ryc_U8EOe3*Nwsq{ngVI;gMp$5nxGfM5NK)T5TQtwq+2^*Us4b2={y7oki^;H%yM_-"
        "IL>l*vjB#v4swztnD~$r9X|(xvlD0}@iR(Dg^VfRhwz99iHOyg<=xX=^ss(d{aF88g;^3hjx(K1Cp622)Sykkvwcsc$Y4iP5fiq@"
        "EF)noxRxZ=3M%X){<$&tce||*PvQkReZ}M#=7h@Ckz_eTOEsOuj3@==!o6UOm~9r*379Lv6rzZ7s-wtN$WrSgiL_;lXtDsxHQd6?"
        "2fF7%qJbuOtUm|y+03h$>}3hUcD%u~Uwg<lr9rk|t?WjhNdi0T!_t~E>8DC>#*J@12)=bv)RIY4;6gV<&5FGwp$g$QVFz4EDWvP%"
        "6K?Twkvasz0kd46U}t$l$7|w|$09))<r0;7M&oL;6N0(bUSSQOiSlfD)FMnm&w92UrG%f{KCNfICWQx+nBnTXhL)lT&gju3xCk!m"
        "*0NS?LA7vJWfcmJBc?Cd=&9%?J*?=h=#(C9KETkl5t_(7YP-^7y5OnOTM`ylN(RUd%CMS_1lvQzNG$46bxc$eGs<1}YUH$hxVkL8"
        "wb-ckJopN;4{2ZX{Gdx!uwJTC@eWNoILAHF^e^_yEMf+G3mG}O=IK-Q)gm%hT0~u`ZQ7w_53W0imTw4PerfM!i<Syy+Sf>E(Wzli"
        "89*@|!}MTk-nB7)oe0k#jvgNpeai8+3}S-syS^`SRyp}Jr`3`X^>QYCMJZIo=nK{@+Ksn^p?4OS;w!yx#xt@*raG2-!GodOYho$>"
        "`JF@KhYY0|dCjr(%DM8jDI$Z5g}B(BWogrRrsT(91L`6jAL4$J2keH&BHK}p_7C*N$hjVwHoci(ThnH>CfqE7`L>HGV=c)3k=!$K"
        "zSw%B$j##2?lL`Ojco>JwuLKjhePE9eBGd)vMQ#P?b7+!;Z-XGg8q4R7=}kw7HkV&WoK;5dZ}TMs1K5Ne(_D4)y3mO(>YjwD)%*1"
        "{{c`-0|XQR000O8001EXK1jz6$OZra5fT6ZAOHXWb8=%ZVrgb?V_|G(Wo>O?FJxhKVJ~BEZggdMbS`jtomg9Q<1`e0&#w^P7kM{z"
        "(+-6t5C#U=8D3xp7?`1(@hG;OXlu(xlDo|={qKCb`I60+flS-jI!EVx=c=P7N%9+0du}c&#W^Xt;i?f*Xd?<z^9NC|niQ=x{K1g7"
        "LYD1YP4s~&UXtCBu(~E%Jdma>d9AY~NfryybgeSv!1SROyZHJ>w=zCA%pBsg;)_bP4Rq8sFN|oVCV})_+e5GN^pY7?)J$tW(3vhp"
        "VU{Cl;aPX0Y5{xCn}%f;yJCeIcKyP;RtwXr<FAr+`p}xiV)4x&gF?hRE;qlao~H}91R&1-Kb-mN1;M9-32UN_a&Ou&W%61`z9vE%"
        "x5j08vf|t*&175AD$uLCWfnxqE0SCPYiiXwzLZ>iO=^VW7B5?`TXEAXiLa~zB7}gd<d$8$`}X3`d>bED7hi0jCre@_Wp(?OE1IT3"
        "i2^pRavx6Wzt(dronEM%X{7GOeyZfzvuEU$r~&`I7m!((JQ8Yc*2CiHna7K`7CrNrzLs$<_~tQjDk4zLZBFEB2w5Id(Ie3zfa^0s"
        "T{;e-b<dq6{kv)JxD@Yrczy1_8upMoigv*w8}jMui_bjG<8yci>dV!*FVsXNU0p91&UdEoYHovu{6v<9(4jznmi3Wz{jNrVIe3>l"
        "ULg-$q_9F*QSi?2@=LOB>(XVKzQo_+Ku~r>@_VlEwC@`@%-5VVwIV27I@+?FVP8(srsOMfMOsB<Ydi=9N{No9>mkI73C+pxtnT>_"
        "O0|k6Lf*4llraXmj}9spPsRwj*GcNO|J8GKgfl_`(~#6oQR-O=CQE^$L#k}Z?be5rbBQ+yqn-(FAr*2Az9ky9%S#$Vp4oL6b;pmJ"
        "nl-zUkqVXMl5k=KoMIW>vXcT8BlWzQB_fBmXLuf!<#v0pEdtG0S<*rFNP9-UZ=G9xk&%NaOPB>ijlP8EDzV4@0u`lg0QOvR1zjaP"
        "QXJ5Lg@R98CAiK8nH_4#`e{%X3p&$)YG`sf0SA+rJkptwov%avR+s}#vLv;tJL{15i9?1);U4U_L@&?L(0e%8zTd-;Rwa)v9Bz}t"
        "Gbl*s4z<}KWY4F7tUpHZkmr#AN_Vs>vNujs=7D5Q>EGI>b`M3h_X`guN&iiq??1s=p{;m7ISRewyt4>GhaWl`jB`%11}Y}^oUWEQ"
        "0i|zf>AadwK)BM~pt`d+tq41Ay|i_Z`eL?!o6e0Gsv%>YMMuZfS`GrPmaf9j$(tfgebWzx7Q8=dc&fr>I}~^56d|!}0fU5ccJ+Ev"
        "*si)>kmR*Y@a2{JY;pK=_v71}oFMU4kGTW=LKebBElRe6od~DY?wdle1jk|J*(HyXp);~Q&*6(b5TTXy4Yj|HC%d<nY8tSe6P%Ju"
        "=g8R-c(42^$mAUgb?8@CiZHxck^l?B-l2((7A~j4Z6Fnn0-tze7-nca>j-u1Id)b6(<)oda@HzX`i`3qCb`$lV$<{_Rjs=r#_PGY"
        "ySFpCC@YW~-(>dlHex?lh%mK28m+{6H;tkf3bs{tV;xFAMw`i=zj<q4Y{-WICZ>Ksdv4*MS+xkxEMI5WAFKwvw%oF7@X%Fued7Rs"
        "euLK#tjm*JOFw1|%Fl-3cD0Q?+s4NqxnW^_-twDCpnc}W-Y^@~=yRWfZeM^A*zJ)?CkG^EJbHnV%yOcgh3+z)(c<A8=KW2W^M15&"
        "F{?S)kHfg(DkTn7=;{Ry4jT3@3C_n{cf1g+4g-l#{FHn>9k`yFH>|pxKNOsodT30$zQlDU^d&!nOp8+A33$matcD`>y(~0#cxF^O"
        "B(W<AZ?$V=zLlLDlxML$?}8}wzdK=S0dnB#>7y=M9~0vg;26gN(kb{TA|;#6Y@VHI-YL%RrpHO#_OFLYgDDu3R5+_DVLlSbQtUB)"
        "0BN{3X1HTs-+UI+DY`c4O?B}SLqK-m4}MRW6~h%}NWb?3vZQM<&mEIBp4`}-`^4Hc*EiAJ{QfFgjyvq&o%foT*z}6knonqB*g%|2"
        "+am{VoBa@PF#XRunMj--^$<8a`RB;uvAv!KbHWpnD{s-hVg%MP<tBW?jPJq#7Vqp2Fj-Bv{vsaJkNl0PG7Mgz{?y{8cT$EieYRfS"
        "W4P`iOIvCG<1kL%HXZ0w_q_Q<wPQzIrd>OrRcLTkoC5v1U;WqH#CF>6-n&_6GTZoh*lB%C$tN4f6&9YnO+9_h2<d!M4TCuQ-;}KE"
        "n@&evOM=Yl70;P2gxL73g`+X6Gx$qyXtK0J^x76r$U(1aUQ7P~P)h>@6aWAK2mk;8ApoMNbb#_3006FG001EX0047xV=rQ9W^ZF*"
        "Y-eR{ZDB8DVRT_HV|8+2bZKvHE^v9ZTzhxhHnRV}=kz;ZnR~ZXBGZx+r;f8~PMWA)-%ahT?QYxacv(t<Y}OR1@)6t3&3C`^07wu7"
        "Whc9<J}eQyU@#aAelr-*^SqbUA}*s@BE_;wlC61`ud6~xco?S_B3i8CqKLEfxk%-EnG1O%=T#{SQDjv<mtvkRWZ-$;mtTDOMZ8*P"
        "c_}WV;xdV6_3t>Vzy4NaY5gOY_17}Hl4<-0JXq%03Wz3&oR`pTAykc5S(TPDABu%sMpaTS;(4iRucH$C(>341PvnNmX7MtcN6Cd;"
        "t)gHNl~Ir_mr0z;uvnJ5g&J&_OBnrPC<>BCxP)giU*~aJ_EKBlmX}!?{$52%T;6Je{+C|}_?-N5c>40^@b$sV?+=eoP7hw4gl~`D"
        "9K8Bv$mMe!Q&`-lWpoo>M|lbz7OG6H*5$2L78VtBoZqS{aaycFN}=MRjPeUvLfN2~+gz@*A}+JM<`kcV@j|90u#ZwZJUag2<(q^5"
        "eR+Ctbg0-B6XB^&iZZX}WtGctoy$Vz?~$$dJ2d?#M?W6Fnuf>I?@pk_2WHYk;jW|k)w1;V^+PI%7x2JKt971iPodIVsPx^bH#Dl`"
        "@_mfThZRRpzn(pr@9pg@pFENCX#DK)&a<UlEOx%$ee!s3Ie$DKJsm9{FIxm~)P*d=B%aH(kVt10m%7SF_1BwctCi+biPOpKEWSW}"
        "-!|*_GQ6DcK3&Skz}Mr^{OKZ!o{XQ&cV^@9&K!83?Mb;9?d{5^pTlq}Z%X7Oce1&P=Xq9S%X0e=>b=5xubFNyv(I*($eqW_XlEJ8"
        "oyE>#w)=E%{&=~!9Pf?ByR)a`r_t`?ugANe$n3{2zj!-6{^8*Cbox4cb@X}~f<8?TPo}1Z7;U@=gnB8rc7jp;XjUw?cDCk8R6!T!"
        "!^if+xR}@9=kw9l*q~CSnoDL2{{d4%_A4awQ~3COu&)|;-oZth<!IqnD_9@JcAVlm5!dmeynGHTr;OqhZAB@uTtw+Y%#v(=C1$X$"
        "7f~L|f-NI0RwA}&hK(O>_JZjl2<PDN`=PV!^$#zP|292#6}~$9;fLwrsjKMljh+bpQ|7n(O*LS@%2QER>qPp^WBgb@TSVbYm_{pk"
        "RtDwmTAp{S#!JCvV4uW?hMVb6Zw?Np!-mws;cMi)dGLCQ4?cED%A}A$6l@Pzn|AS_{d1Y+qQs7Cq8BoiIZTnB<2VaWh|a+g&o>yw"
        "Xn{L|MXDJ`nh{AggH0{fK;F#d8f<>u<tcud=6MFY)dA^Dgj<<<azZAZehj+v;d?(GzI}OoGJW~&n`y`v)f<wQ4lJcDfT0LYaZ7I%"
        "o$eOqcQn02@Dge#l1qoI=aW56m|KGWcIa^V195VRI|?qBEL|rNfP-ZoUC`PQQO<IdX;xia0x=O4uzi$bCe!&PY-(2`19*`{>p-oS"
        "rATDTLM;a3FXHh|S6Z;m{ef$`^aricH_w|UC<28{g5omT*?sDJ??ztmTNdkX&SJ4IV)2NFem)3*HyEHl7zCH{W)WY=0(v^zAMc(I"
        "#rUbAFsOGiDoGxJz<rbREv5!k$EtynI~cE`5NMr`c;u)nH;wt3-b8Z{=cSxq6=Yn9;+C2v@juag9wA!E)~H=klH7_J>_}i^X1C(x"
        "?{DH#2G|dh`?i+y@0CQuB*1P81a!AT5e&pnaS30y)>)PS*o$(2T8nBe=21$uBPO+q(|A>_wosN>wuoUHkm6ESd0dq7{JALPDgrz)"
        "huOH!W5kL%JOSZkRapSm5XC%O%b*^pW)c-K41s(ZhQ`DdGFcAALogJV068wRWHA{9d&8#hT5%_bSt=2}WuSL}Pyh(3N4UnrEKP1F"
        "-$h_d)I99BI|=FzK+EgzR-JCIxFk9cn^q6R8+j4UZ$*}FE#l%z;5Kpt%@!H0h^Pb!(Xs(c41xrNhG4V^#4#|vg-n~oa3nG>pni(*"
        "3oOff2{RwP&tlM!OSHNGlM6=ATZAxqKMJgYAleUgXD|lj#W6TkJS?guXpxbmrLe1gv<eLIbjsosG_R!H(17zkEZBinb1{QDIEThU"
        "Ubbdf#04>kmG&(Y&KS4&9k984tkq0^sPTl{1>Yn%=vE8XV87BbSY0jR+~*&~<TS59wd4(`eVAR*&+Z;l%gy4gJwnA|0JECH)cMpY"
        "C}5Vn2lEp|MF^|QAAk|@UaY`kO!5qN1$gG_6+i9GzBQf<fY2@W$-M{4qjT)~`s1BU-%MYfiuX}cN%7tB(GOw;=m+7S`020H<Eglk"
        "x0A16dt8AV^at&BUwLkNgJ3DkIRI?wJJ?oxsTRl48uxCpZnR;-rXeJL=^2|I(mcruCw=_nQ4H4XA2=2wx@q9`?iA9h4O>GD!8Kk*"
        "FqnOq<yjRcdI3VszR?Tv%xDZw=%i-nPZQz0Jh9>6-#YStVSJ7*2z*f_Ax0oWmI1WX!#%XXFWTC8eWSpoZE2;}y;ny+9-jIS2dsV`"
        "tOV-HRHw7iIpBnG4{f&q4$|82Hb{g<7`TSW_@e!M^D4H8JP@asV6fodNWg9Y5>Wz!9)~K#5&@srLOE9gggUPf!7gBRpgjwm-D`mr"
        "^E_Uc4xRR1A0NFHr-&=X!FOW%^TEmK34yudxp0>0>FOz_`F$>49Zz4LPE{$9^L0EyoKAl}6>pCZz<>Eg{B8QnP+a8Esu174Ir?^B"
        "H6?O>4Qd^2TV(C#npq!v5C?~+)9=A!YBd=e)TmlrJQ#?d4o?4i^y8^GKKkk4^>b_a4mOJ1_gb8Qgbt5RwUo4>^ua`S$HW$u#$&77"
        "-R9Xl07v@xR2&_P<LTQsFJDb@+((UV{>RHVKZ2+2Tb6nFVqm)Q`qN`@h8<BfXf0YSe1>OQlFC&v)}Y>mDR5-&!Oc`poDNM>la*%T"
        "@KTIQ466`Q6d7U*n&AS_dLM2(ug#4E_K?d-6xzVUU47p%bvC{-fTu|e3`akJ=_pz=;|Jfvbg2nv45nxM1<kK$mQpi2153wdNk(>y"
        "<|e>+mhb%qZL9ZpPdp?CjHL+pyCv1T7DSI`l6mMb+UO4Qa|dm3Ep9ij1yeG${M&sxl%EYkBPrU=$d;h%&${JU3;HYSi)->|Y67t1"
        "3g<d!#nDb{)|sw!4)bK`BYqrQ_5n=|y3erwd4Oxz;GkL5wB@U1{U2D&_fq^kyS^7f)##wM^Mnp~Juot@`Ix+DFA7WQ{kT+nIKUGk"
        "48-)n9T|wWW8%;O-PX{B3u!1pTP|FCgH$-W@xi--sixq$vg>jY)@Uo2O9}qT9K*5n5S|tBDo!Hs#c$!^I2e8WRNJ~FmZ+|qr?w<&"
        "3yGAiY5TWX=LGyNMWz#zBPP)IMF!N-k^iYmh-)ibE)j&)k)6?ytYR}jFpYbokqgkyFRS#*3=R}$irT&=`v}`QVyF~7!{7sY`V-*|"
        "RhY_qq0KT8w@oF@M|NG@+qtus&#;ZMYwO8NNg*qM_HJsAT737_@do1i<D(znif@0Z`*m65T7-_W9_PaZAsF|-Sp#wOI0UR8hJk%F"
        ";b$9HG`9+>HO{C*thFUFEwiy6Og^!wWtz!2@J8g@lWRvC>#-J>Q4R*69<t(a1(O$YFxI!#^8v;B=)kw0WSdYy&Ym(!MocLXCoxzT"
        "n_!hfnfPYX0u3elI@KDyc#c7#AUK%x8xR9W2k>va;2kHhDZXrn8fkYE*$kszAV&<Th;@{gag?Yb?<@a_yNYc`3D&%0zPebgTSv#h"
        "dd!i#K}(w3hFHKyZATled3KqS4CXqmWM@Ek8!`N7nD<FGfj0)S0}R{P2xz_I?X=cH$BMLE^KOL<6f7scU`J#KTBRKjO2KpjhGKhL"
        "><!!~(Xmq7T{tn7B~L@kg`vpe&rHY0Y4xTGm0NAMnQ1lMkBAae5L>O9-Eg5n^qYyLCyw!HM9?#p8Yj@nf!5dXH$FM&GrBDVQBQDB"
        "-4?QFemJTdn4^Um8sN#XD;#qRa(;@V#0T>>g4-Azv%}2#NUT0$4`C1S^XQt+Ab8`}{H^nBwJwJ`c+YwRRmqNfT<U?w){mNs+TB;s"
        "DlX!*C?gQMuZhBv7-00rDsD({74uZA6W(eDL1{dgLPCe1h1Gp71BU&%cg%-2Wc@GV_c9fn5P^e5jea+Sg|B&BCecMPfz}6i@R6lD"
        "uR?XaL<K`>N$`%MoVxz?*I)e?`&{q;*0+Oq#lv^0A3S_9c$Y#M{QNIh-zUvjm@4QA&Ls+uD+gqSee3|>$Dbx*=bloFR<rn`%Bmua"
        "({)vbL|+eKCYKpGJoGfU$g^tgk10t1E>##Qt4e6c<HTJ?OBu#ukM>1(xSpw6?h&gF)R2u~;RXtL3;==3>sm-)qFCd(5yc;qB)gW2"
        "A?&g@99vAHlmbNCG@km*HfXIyk|Dzk&6w>G+)JL!cUz$NiIyc<?1ns-Sq5HLV=`{p<C4K$g_k930yrEzTd<mPFcjyms;HCI%-A1u"
        "@1J)q3p|=YoP*`d3}uVya|Yda0ZnK_)H8n=6^*CO@=^~6_hk2{npZU6Ij89<9HTbH5+;Q*B`zDiARcjz0dHm}s&mS~SZf2SLfI-0"
        "5HK3i&f|+X1xRM3nt_BsUTC}maYzUa@h)b}P%VM^BudG;`<VwDU$~4D$}TFhq#_KuZn+GgVAprnK{|+|@aPWe@#3|K7!^pv%yL5>"
        "xG_2>Ol6WmxfKNA?#uv^XJk!*my>4K=_e#tQI$rpAttKO=x<Y)&@!Hfn0FWAX;WCG>LhUy7E&&(959$BY}k5QtzmF7TH)5`X>Ya3"
        "CaSYW=v*RPM=>H@R)j+7q53t{rm}U;q@F(1rI<AZ6o?u13>*U(zB<zHRtt-yf~DbWiP9>RMXNsMFsO5_CKcezFN_Q?JYh#eNN(bD"
        "c6PR#pb901sw@bC^q9yq7jJpt>zvOE^y`{LF926^&1c(Z=US`L-Dy2SQs&Wko#AOchY3qbz~)0581lx^jdlWwVF%2FiD$dcjEV76"
        "D__DATFdOH2|!L-d87W#P|p{~O>fY!v}RRtj)kllLylCS2h5scK|Z!BD)q|9EWLBH;g%t^3^94R1iA0ic_A0vGeBfz+(M;I3Aw)I"
        "oT~*x=Xk)rOsf?oq4_vw+rd+D2oEZI6oeRC0F}F3#L-2HF6z86^r)U{=Xt-O$`co$Wj-eg)EaW1Ow1fpj<-PzXSfMv(@LQ3aV#5V"
        "hN_#1nH+5N4gk%WlFNK>rF#X1G`h{M&pgy-?>vy{qPULBOW(r{qVFJj6N066;W`{vmYVV-RlK-1^5=T*3=nnYt`mC9crz&|-S5kv"
        ")j^3KX}c#98@1V-G`o*Or1usNZFe<gbM4{MGrM%2s)L0<rS9piZELh78=K>9RvRM3M{6+E0K3nI&(FOfPYrK<=RK`qHg4K@$NaeS"
        "UJdoNWX?Vo^mlK8YJaW4QDL*)yR?t|WEdmXABaEK#u~F`oJVF_w0J83<F>xHjmU2cJS59{^JYr|CBQ~9g>L>8g=r9hF+!CB+Lv5H"
        "AQp2=X#N;ddFN_639rF)_NSI$x+i-s3s<bDpVt(_gR|SxDR)Jjc$-XGylm*iY1Ew;IyR&y;fHf|5&Yufnk>B!tc4%XKCoH;c)tGu"
        "nfeON)>r2rdxH1)=@}h&<G8Bg+xJo&!|I`_W*P=<K|>XXunwRsVXM}kGKrg|Xs9DuGTi|hFZKP5qiN^pVz!?*@iK>5*dSYkb#(gP"
        "N!50f@hzXCwF(A9+vRAj3RisAX+wx9I+xJE^WIAuU@{nbgqP;IB<%1cu7=<l{@-PHVb7n%DS?&N*pycff6OsC;0`7&PYifmD-+Z0"
        "K?qR&{xAS(Hv_FlhtFG@;ONWXixOvex6tYYdyTJv*idZX*p?$Dy!26W0=E9cEBy4tlJTRkHF$&uAV1@W=Vy5N=)5ORY_txqB*lu^"
        "3%I+waZAbBYhAvv@2)rPBxYCJNYlhTK+ps^_iYw}4T+biOX?n0mGTo<IXaZWPJ2JWKGfddf?#yZ$qyHDp3r#}VmG>TkX2=<dFeYU"
        "pz(ZW6yiTPpbO}eb0s%2j@9aT4F!zvaDdSGs0N!A9mW@`xL}Bfmq5SO_>PW@7>1L6o^hzF1pAb39O9-xj<dl$0!Ih*F$r~K$PCIj"
        "o3NE`HlA+q%i5?m&@wg-G;UgW_<d6|6^aQl6hrz)ftto&%av2ItkOBBlZSNj%*$>QWyWAO@Im|4K|1$gkOI2U`b-ms0b0a2fe2Ka"
        "PQQ07i?cpebzHP_?-ysDK70z%#D5ZGq2_Aa&`8}j?C=~X_<j&Yhu4B=y#`+%J8&Wij;ZL{&cI4t@pEwPu;#>d1w-9qXpqCEn+vc&"
        "3ku}p#!Zzrj7Y~4X{YCw%DaWlAJ5tQMyeZX!Tk%n|J{4{C)*t!37H%G@}FIjcWXWZVHHs(|Ggcf`){#38ja9&j~vsf8}>=KBS(Y;"
        "6(2g#H7Ncy81g>a9;2n1n@2*yWDRccu9ZRE5$d#RTV2_q=k{s3I=InmV%R#uQ478vCG}*UH*{w%<clzf)9!eQf=R$)*_4;vrCtuC"
        "a{RS@7}Ub~H8*^#Mc8Kq9HCz)RUxj8Ta@aKruc>e@WLNIZRa%UyPS?|l!NE&&UEg_;$6(beHWDa*)uL+8rdXty-((43(py6Ib=gZ"
        "b#sN{J{Za^eD#--9}w3}#$V$*INk%Oi)b%}F2}7Sm32n`3jiIy5>Na77Pb{Nc{<(&l%uZ(gJ&J%ZUDQAab=#Pw<GbFR)^LdpRSJ<"
        "r?k0;W+NTB;k$d7-GkYq&23#Z%)y|Q3Jn|F!jH|6()DEw@@6V$@j1r&V#@|Jg%ka~eq?7HHC1AxNG+yVvw|+4ar{H+f3%lchYM;9"
        "20!3yQD9Y}ul?GG?tnIvHEQ(;n<4m9JIy+42LK7rT4>QE@<6;|B|uHb_r*{?99W1vD=3zDRcx!HAF-;6QeP4-GqCQLG7nt*uwFI8"
        "(16;>eEqHxs*m??Z~*w1`R9}zR(x#b4FN0-3{pmQEVpl$E_ET!k82~~nN>=J>Ofzy_8GLaLj*l3Y;|=E5OT4xc-j|eQK<)?tRxM&"
        "n%6G6Qupzjx|qkfIgN&sJ;POWCBd{8n86I&Hi}!XM&Lx|=A9iEQvqF)Dtyy%!2702w3;m<&hT|1QWP{DH4)|9MzN&KIz<@PeLKT~"
        "K?X3kif;Td%CQ1*n3h63Y*{58VS{JJ@f>+*#&7CWoqMXm8zNjt%s`D)S_e{6Qirc7Zp9QG;^?QGT}Weq8Z8F-wHe5Pvx@+{PBq6x"
        "Gs$RinQN7L2EA;YWj+^eHjdT=(|IpGXpVb*U;C}x58xc%gHAl!RtBbGDrxoPar3DjY_qCMa!l}B3z3+|;n>~Kl*N1FmxX-2X3M7|"
        "5Vka-wwsbtd-a$bWwo72&nVirSCCzo<-*mJY^)k6r~P{J?NW-X?9(>wXjrSNI#@usX*@Mrd~DW&h6i+wJ<2gugj2y+Th)Z9OF`>f"
        "DyKxKNN-(;hTvzgk}EvmSw(A1_2uSxS;ZS*eclK5&8f1Fv-!OTv|k4CdLCqJNpGV9C&TZY>?`*Rs=h>)4j2&FIiu7n0BK$Mqrp9T"
        "!$X#CU$ACihl@%W6}KAVORZK4K!r1mK+EoxXX}k2+%)Jmbe(72#xhy(O>)%^9U^J~#ycppaD7YoX7dF#WPuv-fgTdt+!<AFlh^uz"
        "vI*52QcT_8M<7V+)hh!-J?IS#)QIHld-;sj;u0)fA}xc!*J|*(8>VdVCa|xn(z|^8@o47>SZDl$+2USD_RN$HgY#9ImeEiZjF-MN"
        "BjsFbaPmkUbTbCY{EOcjqXSk8Y5scHt`jDJ(96lr=*eEYdbeN}6_Q?D$0bjmj&zAD_bFOnN(M!`R9BAlr?)%^^&XfijeoBw%BeG3"
        "m${UGU>YOFMmfuCTeAThG`{YqLF?UD&VhYg#uwT=On-hmJwCvfRfQ*Sr>{=>T9M&{PK}q<neXb;Z&@;rjamHK<TO4gJ9Y(K*}5n%"
        "Id5jD{`(A=lv2)Q=SJJod+^|aI51K@Q$BGuas<CR|M(7`W5w@4<olv^CaW8;+}nq=rl~3A3jKhG51}@N4TvKOxY9Qx9MH9(>bS9A"
        "fUet2(4!-WV$bE-$Q5L4uLJYbUK5TsO3BBM$D^-z`Uz~RW?yyIdiyyN=nnV^=n%hJNRP$Mx|7q5j26II(SA*w{&=bj1p#nE1Kn%9"
        "=->%zLSb~pgxOB>wa(=-zM<v{OH#24veDx$F?La|QG~0Jt3hqS+U)4Z8(6BVY-Y=fhQ&*V8NB1FRFuH3l42x=F93B3=$C@2B{#%A"
        "Ld@HUp$@<FizC9Wvs;|oGd+#rbmv<}t*ytd0|2|>6K+^@dshppjjoLSc-YXe_*FJjdxF%k8zpt!sIz<iroA+}Ezn8eMlEf{maXNp"
        "sbxpGuyuRSw3eMS5>Fe(=ZDBm7jxfnc5<%}xVo-|@xEn}U2i;}QnN&tul|sqAMox8KF|o59o<u{u#UyE@oeVaXCl@te7IYxhVYs*"
        "l#gwY-iqh@F};s(LnObB;(zcI*2BK38FW{lE3paFKkGh!uP^}He)}2w@+d<K(Jl#L1N>0Mu$x-Ua0H|YliZw9oAV~eQ8l$@wPr)t"
        "cWx_5G?U4`>g0?<CpaknO?DQ89d)3?VV;l;qB5+jDOZ2JI|9dk6zr0lzkaq$zn<xz&-mvv`U&<4evWO!`LPAf-6ssj6qz9qGos%&"
        "Wg8C#T1N)8G<6h!tP0ev3)<H^ps;AapO*@nO}u3mZI9jBxLbzMuIt!Y+osRq+*8A%TH`yM3%O8R7p|ach0!u=)$0>$Gje9j&^S)u"
        "^otO~GtChKU)CB@3^}haQRZ1z>i6(*jX_he`IUS0P8}U;vX)yNdg8Z{>6jCRP)yseUIR#6Yzv>8JQ7}j?<!Bsj87HIC@+<p3YM9t"
        "5WJO%SCz}HJrCdEb&YbLV8)u;i`Cre&=!lwVSV{kBX*EWMY(!X)*gsself9eM1|K7$N+E-FW&e<>$!*$bnE~P$yt_N0g#DcHuJzu"
        "7>PK|hyJivvSF{XZeX>OcsI1OE@qiIfOmDLE`=Ev0PgDO-Ypz^Dt=TTf-VdCs6xS_TCEGdZvhJoU!5Km^Eg(@t8yF73nXN)NzX8q"
        "jL!|M=nB_fMQOa`@QP^+mpJn!zpuP;tV8t{Ia-t9`*G{S7xf7(mbMk+rK-|^^0Hk-aVo&#T_eXDD$X`U-C65N$9pvVyt`z27j)OY"
        "eQe9<%rnjg`jS|5Wkt?cM#0n1$tHLMp4|}Pzkv~r$=3CUg5McF*>TQ&*L*Owb}&j4`|^wb2T)4`1QY-O00;m803iT=_*tYj2mk=T"
        "8~^|$0001Ua$_%IX=ZO@VQgn*ZEayMWMOn+FJx(9XKrtEbZKLAE^vA6Sxa-<HW0pNCjSAY4<QqYcHB;<RqJLtY1-+fGksoU4g-;p"
        "g_;Cd0JP<}{_oudz^5o#?&DU@Bo@it#Xdjmf<X}c!nS5EMR7+Iqd8L~=k!WQZFr`?AT+0yVL7RkG%}OLj0koEL~_068K(tVF_CRb"
        "s;-I7q+&V_g5deH=g)XqNo7DL-aLPnvo*+I6;&*SKk^Tn8mg~#m{a)6%#0X$%>@6z)Qr5Gk&<rHObP?arS^@!_~h*KPou^2X9TXQ"
        "UKKo>fk=Ks4I>w1natpyd%Kp3;39+z;^i-c>dtJW2of*BN}<YV%iBEz!+;#0-am9tQDQ~SW=5=1!k}mk2C_WFm6m~j9wae0tkhu?"
        "^%lZw<Gg!8*1@GXIXNNU@`5eMJ&Jv}6v2?%tFgql2{7rp3=1a0DssBu`d0mjydwC*iI0X-EJX}HWFikeNozvXBVN}<0jmT-{JZ3^"
        "ij_PxpNpJrG5jnq7%tC5T`~pU_a1O>bdZcK*Roa_#%ScJrQ?9mfRRCx#E?odXJJsA_58D7WXp<y7AVDuTP{e$t_`T0uI>y&AvrrB"
        "M6oqfuuM<RloZkl2hSZR+VX?gCUPU=e3+aSBUZk)0a!W_3optsv>HbV`RJlEVB}+K9Oask-)K>@cS=bW20wwrfBd=$fUc+@CD$c2"
        "*(M(A&xJOs&J33VtO$3-gEXT!R6n62$5=9Ih}kg0wlp(ffpE+@)0yHc0F4zZ<n0VdO|1kMSH!{$hw4H<#yd%m*Q?g^PELg~gJeh;"
        "?J2DQ>*^==N6oYW>HCDibEv_c<vrezo|kvxG~pMFQVU*94tteeH9}{Aq<N|<hM?IX+ApeRQM4=Zs|olUIp8Vb&tzE@3{cY7F571C"
        "L&M(+K$M%-@Kv2(F|(T)Ya*oSoDxH#@G5+g4y>%qomXIIfr70~3OU4YEK2aJP45<+v`daSiY2qKwUWddOfdx)Oo!JDQd8vKy`%a^"
        "*$wY>nUTfO(oV=b@4^{D>Nzd=m4L(thqlG^1;LFDYDV8eLU0X*hVTb(Xo1xr8u!OT;fT382OSZufJ#wu#X?97yAPlbqM1`2ZHjjG"
        "8y~s8Z?SG$kzbNl&IBi38PN0GZLdhToWCaXRzT7v_cZ%-wRAx|ST)VkPpX^d`tst@0&1>(wp;sKJZROKCT4}6!sE`=etLgPfqIwt"
        "fQ+N~!P|qvnTWMZlp})O82-wMT(%+qKhOsQJwmRhNBTGc9*@VBZ85qhV9y#WX!EZph3*OBzrVlX_0?490A4;m-=*IdyFI3yke`jL"
        "i0@7WgPB9a<|b>AQ)pmds&O;S?{!%bX{VWz8=P`1<j0fFX=@1-KR-J=J6<K8BiygKZ~omJvRK^PZukZM{~~#Sd*7<O5VU0QWYnCv"
        "i6^EN?nV=lx}Vl3z_t$cPI(?Idb-T^(7Rp{NCS35i>j#gshih=!yGWx-EfWV6|kP4oqsYv`*eQ(@m?+^i*}X0M7VRNCFC1YHVr1M"
        "Wc$zqe7#;4jqdLDG59nD;+@1k%#l_S!g7cD`B6Fjj+G@%T`!duY|ArwrD(O`S&I3U-ohj$Z*_X}dawS?be%S1H^{eOFv0LJd^DZ_"
        "-GKbx=gr>@Roi$0U;F!LGu>!0J`UJDa>$g_8iC87Iw%OhzklQ)Z*8LnvU})Ym&UeM-o}s-VPGGuVfb%W3ufg0A#yMMeb8@_d)t$h"
        "Wv3S$Za^f=mkQc@)1Q!R*EfOLk3;q3<S|0J-4_2XYV{UYcn5(74^*|Wu)2on53p7e4aN#UV+CnJF)s!azyKeiQbTK2$qdD_n;==b"
        "+=4tzdR&L_a3ZIllFvO1q$T|z#aw3`)ABUS_T};v3TDvYGnH0cz)X;0Dk_;Vt>Fg4KPl!hSRB@w#S=f?;i6LvLo5v5Hp8IX8@I;J"
        ";&|j>Scy@hYXgzMx4N!iy0M3UY|XbZOxCvA>=cg#m?vD+Hl#Ch%VA@yjXxQ{aUW?s@1$xS>h1^I37`d$XmgEVD=f%(*y!K}o~K>9"
        "O)h>c#`b`xpMHmM^`GY7lVdypQLLUTh20K#0MDKtHRv%E6;hQj+I|3kTOh%BG_dHA$qumoT)-ouKW0-b#%bFgI)y5D|NedWX5pHk"
        "KdZYxt*kef`sJkv<CkxuO92#sfDWmE`DZT=56MOcUw&=-I7XAJ@b%Cn1WI?LiEfOQ*U-W>K0&<htKOgdv2$%aV9&PG-pP1~_^HuX"
        "Ea$ekbmI{ZzxQf62zui72zzjhyblqc=OLxN?f1C55c^Gy=M1>*bUN5`1>HjF<IDDR?G~Rf;0Y@9xLZZo7VN=p`hq;_1IMwtgO^j}"
        "kj>+5aoU{L1sbTWxCmy6Mw5ef{VJ}7ycN?{k?+6h41+PEF5Cx$PT;84M|D0nXjF;_Wg5@IcPY_AYsi?zp0=j8oDQ)Kx!im>tYS;H"
        "91BPya&fUk3sfD(BvDVN(MOUlw(fZ{D94_&aD!Jo>o?BMGTWs&^Lop4aNT}zN^Mb2pmsoEm@82vyzC0BPhsWT_xNE2tm}KVm!&@B"
        "1Ku8Kil0A@7D7&Y+Sp{pU-mqJ{2;}^gpz9wTWH!#a=(rp^OtSrJff_BW}9pKtN*g#F!$0%c<_TH%?E=!Vd!?Q;e7!kv%&2i_&2i1"
        "XJpv!wT;6mG}@Y}Q`o9(AJL?cl~3>kQrG9X*z1`53Z~Z;fG?w^16k$1Z{b#?AwqNe&>mrfBHH4NHvuy5FHlPZ1QY-O00;m803iSf"
        "eN`B4761UiNB{sL0001Ua$_%IX=ZO@VQgn*ZEayMWMOn+FJ^UYY+rD1Z)`4bdBr<xkK;Cy-v^lgpwr%A?X2y#-FCkMe@F&BxIrG9"
        "ok?~V@EHOvQMRUKNfD)fq__Y3RTW8zlxR;Ui^B<+X^W2{S??;AeBb}3eoWGYl}}!hiK?u_D#<gi$n$jM@y8-f!ld%xf1U~MiHcRV"
        "@K_dkTb_mc11s;mUu#yftV%LI@qPd8o40S0Ly?!2xBD$AnwQTa&zhG5tM<)H#mYEIdGlVEX_{;&CI7YNqI&x#F7tyI=4r}lZ^BdK"
        "-sN>x@lp*a03=M)oj=hEs&8NKc1gC2S;&KZ-Hh4C_!dO@Lzd<&YKCu<I1gF6<A(#AM66<yur3+lMa>+%edEFN>94<he0LjsynX*^"
        "BwOC)5&!EwFH=^0$RhrznrS`{3YL^S3PfI)As4C-kPILL!nkZfYy?rVgVU>)lCvmaWtl$&@K4VaK%FWG^DG0*0Nj#?c^T>6D!=1d"
        "P?R}>gSO$()8%oT0`&qBSNecJ=q+p_2CVV2D3h!@I=gtP_IVZn7r<_wIv5f)j4PAhc)+4VQsFKiSeXG>W4Fh5R`LUn5)1XLj^NOQ"
        "1y6T#4eSJ<A{R-Om(2zVrIHd2v?33ZhyyEEPsj}4-v0RRzkm4V<9EUP58vN@@>brnJe9w%4=m&NERB;#|3)8{y0*zK1SU9Zh7>Hk"
        "^GC*rYqpqOUd-dmb9NPO=jYq)@@h8QUY<uw#<sJI1&_n{{PN10B(Z!}0Z)vl_>A+Ie#|BXD?w5!J$$JC`XS2_KzLGhc9_*gnV!94"
        "*@x;Dd(CCe!|7#wezl#&*XLKu_<TNF#35(bi)e9iu?R1(qj(yfUodMDV)%FMtmNWX+?4;fx-0?TXCH4rjsND4V9!&2wVZBe;q^3{"
        "!OQkCI^WLY^YEIn3m#6d!+AJe#0<ycvVvC=)GR*2elbb*%g66ciXm}-Ww5Bfuc|`aoSp5G3Z#D$=7%#PQ(B8NISRNb<pBCj{S{|g"
        "MV!s2^Tl|2F`l2RvE#OdHZs}$<~IZ=b0Lr!5Wcx|$sQ(h`C9N21h@hu&`#Ka{g!8A5hkEs`B@m^)U(Ajyts<y+sly6*%fE}dKq0$"
        "XDqy$@(Z?HOxe|fEzhGfqJ*_<K<p1HcJfFYdqKv2`zGSC7xOCI2aO&NP)~?KndjBW`)VZh$Lc*x1s~ohH3WTtQ`D_AZrC<Y4tG&f"
        "4uIlt6tVibtU20`L{ve3N3Uuy>S=+5c~QzPFTFedH1Y<BP?W;^M1Z`HNDXG}z=vR&ypAkRfU^%`pf}b4UIVC&-eTK_Pe<E#dZm_y"
        "4fLD!J{(=7GODM7uQWIM&QbD6tLAL|F(C{0sPxMfA^8lLC3eA*RH6Sny~Fl(*g-8=R2zpV97j+H$WuFjS(B1%%)mbbftwB2<JqPi"
        "oy1-S198%XC~X6y-Evg6=v=JixCh|+K-4kl%>gyY-`-z5AGJl5fS?R*7`8hwf!C5(Wy0_I0Dh0!HHJslM&+M^>8AyUM;N9ce?d~K"
        "s>@9Fw!*TllQc4fB|+;>5W#YYs*aw}5XZ@mWJiFdVLLSV)qeG3p0yGLr$8S7Rs?1lbhvux{j1ZAgi>1ubYmYaIrRQx)xa`vUa|zd"
        "w7;{o=C|N2lmq{pGzA8}=Z3$AXa-lP0za<g;EH9yx@Z`;92kasXs88p>BR5yk@QKBm7Wdt->TMdx}1WB0^P(eqxkANx{ktGJPXga"
        "pv0#aSLZy6=2N!3m|sKFG7cHzi}Q$wv)MAbxSCm3MHvZyalQ?&<LSj>!7pMqJzre1aCsF2aK1f{FSnQT^W}WG4B6$DF97~!crlNb"
        "OCE0H^Tj-xYoIR<h6kJ)&eovi2tu%I?XYwo#Vu24%m#ZV_CdI3;5|h*K5#d{)MP})0rGb}#eu~$=8KvL?iJc1<Q#+D)`b2*b^1y?"
        "w`m^UHC$$oSA7AwME*uW57C)+=w&4K!1i}WrfXqh6Wu6)kb){0blocxw}uRY^Gm)fOasRU>49smRvj11<6won*>q`<4%G4hVH2@5"
        "q%V@I&h$`2oay)-O0{UwN}^e?9e`*`MtF3@6wm4dM^|4C>+A(>G9t)fde$(0?Sr#^C@Oyg+o1O?#4S&N#k58b*x!si=P;l~o+aS$"
        "8bdezHI5;-Ka#L5w<MK9vown#;_5-~XkNt0=Mf#(MhYRIixt`$O2C9g(IEDpVUsU6&k6kd{NhV6Y5}w!6je_00!SfQO6wokqF;jj"
        "h4o+xL>Mc-uHx~PKlGUJKo+Bv8%v|T23nBMM*e^tp<Nmj4(P^FYvp`v;Ub=ZzG*8ciHsgPEPyb~(<;fMb6UX$y~%^%F_bV5J8VWi"
        "h^M?oTuRPT4gSWV-@3S!sFKJ<a1y^Xzq=UcfQ@z)`40M`$J25<tL|>rat0s|s017{A+qp+Cd#o*I{Ytl+Y;){GZEp%<sqpZH0whs"
        "3*q(lsRAzcZam%0vq&{=CSZZgtsQr)TJrw|6jo!GWfLf&*>rl+8lTWTii<6E-10Wymzq>moz^>{l)EV!DwASKhaH4$NElcd?!hN8"
        "wtC=)EiVD2>YD6$CEEr%^rRF4#3)6|QN;x1ZWe40^Cg8nme(dK(EbMl7=NrvFpIdifrM&|W&&cMN>54zIhE)$b+m9RahgYQN64Xa"
        "YFF`tMmEG-3~+RZl!!<2cR{lqc*RiKfph&fTi|yz-JiT)3bj+MW^l_8Q}A-!P8*ATURHptLsCt=9~cB$UYxQD7?Vd`h@ifs1eWGX"
        "hbW~mBAVtkiuye^!BQ;q*kPOO>bw@Pd+;(Q{Syh2qacupjvyGodHf-G+S@_tkJr=jH5<p9=gZMz`tqM>&EX%;13Wzdy@$i!C7Tw1"
        "X}P>XQaeK-m;Kt6Fe$3a8Wn|XME-+)C%Zi*xGLZv+UXZ$Q(K;Wo!#~9v62OH_%zIoTnJqjfWHRc<O|lxeOy{l+bjHyJS2mZD4>SM"
        "(bCT;H8Q`rSbE6dAjU*d@^BWHv)3vYI|TrN>75AjGH4l8DxI#j>F5eu>6EM^P!wY~HW-}s&2U0)A7XaUv%+l{12J$0S~-J|41i)-"
        "5muHTNh?v31c^i22gEiFB&PIk07b3Q1&m}JYDD7je_KbxDUF~=d3^x+bygmEkeVQHJz?lS%iKqgf#vBBfYvyH-BG^A&9yd?DFH08"
        "OsV@$mjtT~_YkAk@=IcXn7Dby1N}LP{=Ct?8iF%?fSxf!CeeO0kfZgr%Bs!P;X_21SQiik${%KQXmhy1*d#A7MF315De*C9TJaCg"
        "Nw_gy^;$EUuukmRe0c$bpYLw0!0~=4<E4>zk5nU}(C*la2`b~P3=U`yf@%-=*ym{k@FZ%R7IOw1vfvm8^GDdi8D)OP2*%50^yua2"
        "qfw&Lyp=)RD6ceTha`P6?rFXTY4HlwP%Ob*Jc@|m2kZ>Ve}V~_%({mh4-dfwg%vuKwlXEO`H|*(X-iZ(y{y^{ib~uYl#<sV#Dl#g"
        "N*5Z~j2&P{SlJZc_utoP>d|FIo4D{qSaO~LV$!FtJ&~~juv*D$tAcE?GB`;gfNm++J$Z8_4<+jKcqA+y@K)pzf*>$Q_)>Dhd2m0s"
        "V&Nv9R<~3GL&nu@+L2xzdN{fSp2~9j^#1ZVhE`V`+Ks~r+Z52Qr+vlIh$?$PQ`}f5{Z!?XOEbi%Gg=juvTRqg9Y-9g*l9S_?uFFU"
        "nfI)_XpsNnfJ^A;DQF0G0}a)kl2y@&L-V56J(WIj0EENo*iNTaQ1wGolUStms6X~8Q%wx}>cg}qhBl&P+I28x<5-$H{4g3|SU*LH"
        "JEhHu^$`hon~6ZaA2=4khj?9b`W7$}NEen!gn~mq${eKa5a3-fKk(LzFdvuwa3g?ZHrs(v@iY=HSR&bhGO9~Xsju$6<UrpWP`1L@"
        "bCWHhjc>YMtwL!F97zZA_3DjBHC~S6<s(KG$nCDZ14J^J%{q=jS|6-!A(E|#a_}EB2r)p2Trx@9L#46C|BsE(k}VO)(`1+651HQy"
        "w2}<es=<y?yxfPyA46VL-k?Rr$oq~z(Ju-vIsS5BvGoEz%S0V!+wjCR!Fe*EkJkId!2_x0VBT!#{g*e#TSm-hO&!bWiGydqXLG=u"
        "Qa;;n91L`h2J&tAVku*?208)wY_H><-w9ECrG3#LYxFi#^m}ho$2+0HGNIHRNq$J<nV4RQt?GN*qFN60EI_RU+7$@UU_1qIkR<JM"
        "1<6tXq|gBr1UEcH@4N+WI|mAZ-HVhM%r=!g=sE$3Bk2mL9JYNisjDR@i&UyNnYht!mIoo5Abc)rT}APg4FgUEdSW%7F0O2F{SMSC"
        "l>d-Di;Jo5vHTL%2J)Vuy9AD@T1xk7wTOryYmC`ugLzT?(j^=!BLgKw#14}`p~Cd2-obg7X;qBKyDt=|CTbh%yEjY-PiUyhvpaM5"
        "9lbR~dg6~G-k)vP$h~)8p&5abli)`PE1QiQ6FQe9_i!Swq45?~SMJEwK$ftvSxHR9uPONHO@)F!l`c*(oXVBEipMy>dO%vGA?(m}"
        "9gSL{jd6y=s1hP1n9<2fW7XX+W1W@6csqn*41)XnF@twHU8E38O9yFn%6Bq`H&UU0wD`5KYt;7yDUZOW+ilubkVDt0^)V3N!0WAR"
        "Evt<ljiU&AWmic#K(;Nm=E29-KE@cOCakyqJT=OCGbwHB9Q*YZxF)joHBrCGri~0m!!#FMJBef)MzHPsk*qmGN6cM&Hqbae727Mx"
        "0n^hjpfd&pwq?VZdu`p;O8F}K%5t-59a&=uD-v(DO&b@(DqX#0)nJkA1dU42&Xd8h0HNvbTxN63hJ!v$DVk@x>@q;_H&us4@q^dS"
        "J-7+I>wy$G*ybn758Tt{z&6V=&f4A?2B>`}qXNpd)B`Y>nrUd*A1}ob#<`@lJDehZ#7I-J2hJEV6ErHO@`0<+zv1^U=5iLP#8XzW"
        "M<6@fWx1%5Pz3F<@8<y%mGVgr1!7TG7{(>%LB&dNr!5Z!Tqb1yFvlc)prETs!hvJG<u4aA4rpi(Z}N<<m<|JvVu3UjZm%urY%zq="
        "h)dvKYm#>BAW(I&5Ab-nN1(GF@xrh^)T!pXMYn%TuAqS9xMtS{UT6sE=g}fC3afr6Ay3i)a+yRG7@bAmHLVL?-b26@Nxw8`&JH*n"
        "lxv9#uo&5{g57U0szECIQtnHXdp9Kv$bjoV0{0Y&4xmmOzG_)iumjfyg#bYe4}s_7*WNY<Uk^3Awj4rJH~QN9@Uyf|fZ`JL(w^NX"
        "I9EUfB)ofGQWjWU7lkbK80AF0+xfR=w~KErhHtfv);~Xj^NuAbC0<{l2<$l~|I8Y2rHn?Db}$NPbyR?D;0oE2RcE2HE~+3n&^v`y"
        "shH5tAs$$k#8_0cG7F(4b3mL&Hb*(}%9tq2Vt6$pOGzq6LSIWx9!l_K<eh}EQGfC|PqG1BXGHa(z{1CeA=aV&Pv%?6{T}3Dzr?_{"
        "pl#G@*4F8G)aJ;JKuVz`L;#B-Dv&uLNLK!v<4UbjPpwe7MoArnV>%q%jBv`V#QD0TNI)_iqc<I>FYRmN4p302iQYzq2Y}vGF6c_{"
        "at9VyaP5=Ri@c+j9g}a$U5y#2pQr~8Cc-i)kW{N6i1IK9hUQdE#6V0W=M4Pun2Ij_kw+Dr$WZ9y#0&yf=`{Ni>*9<K3G!RiBEZz>"
        "@HAXG?bU!=+ycFW``MSkH(J4U09_Sm*W5bckTXA)&itv($`P_UTIW%4s;ay`v4|mX<1y76!#a#K#LCC;4FnUm35|Q67AyZ&e{fRN"
        "TlnFrqFPTE?FzcNAT{efaJ(HB6I=L>9F<%U<AUD3jngociXM{#uJM&*Rg12x+4cN14CU;ffPq;DUGY~^m4^aFe-M}$w2Fo@!q7A!"
        "<et3O`~}_>5x}m1Q&xeQ0xt531f2NjD9W2MP*{$ZlDz9G`IU>l2lI7HooeNeJg(8p#bq(UO&Zx_DxQw$W1~rX7mdtj-!#**X6X5f"
        "duEzL^6{$D^foWkC~7namIB9YbH`weyg04J-V`=`2JPw?Rcnw+U`&%_!y8ffl>rwxj7+o30u3+ltf7?*No$uLbsDW?Sku~Qlb}1P"
        "mvHG`;jIWjxAkZ|Oru{H6w}aSk@1P_g0?5eQDh#{LWmm=77o5LhptR-riU5gvOk(`>?ReB`_Au^E!EQllfeH4_%IcDE8xF*<Y%f1"
        "E$%hntT92o@4o-g-g5RolTvc0Z13$qf4cqn;fLEFe-1wVbo=hpv03u#5{WtMGc6R{#`Oyt+-eq<qM^RipqHUZ65XS`9w&kHm`-!6"
        "tbVtidZ+Y}+Jl_<wRaDGEc!ZN|7w^9)(P+(?(<xr63h652kHd^U?6*|#a^v=)MlG|4sgej>NgrwUs6l2RZYKlUvhk#DFt-9KNxDX"
        "Tszhunct}lZKfVP<JH5CqL6;+ypU9ke0>XJW_Dbk4V8c!JI&*$gKcJa!(WhE{5~(n;CZb+WJ7Q4*q3Zw7uT1UjS_~wm2N<(%1IIu"
        "8IJVkXJ7<K3*fAnrj4JIo9Qxo(W_tR<UCax%Yvh9#|IZf6WR;i5w!2U>jOCc+UQ4M`pYJSNdr54)q^o%MZq&yA>fz4t|GzX@!C(K"
        "9vVzK9r@HsPH#S>iZS~v-;P3i-Pc+8v|nE}-0c4R3ZLVjVByohCox}LLZI&lCEt&9Lvu+<wUjpvZnR*iL_jMNI;!f5i6}S*_C1~6"
        "xpz{6G{E87N{?u=HpGr9Gn&+v?8-U=9#3Q@B#Csfhuo~=npSqK%{iM<dlb6^OUK-H<v72Mx_09z$^PWsLg)nA9zVUrQ_<Zf*aPXj"
        "V$$YlK1L=<;5T^s>QYWdDVQLew6svASX5v^H6xu*Xi^@d@oUfT@A7<?@-s5G<Hh)P{w>~{OE?gQ%OXHSuBm4BG>F9)Kp>Y96Zy?~"
        "rRVzM{i}P|XpGUmISs|j^*yTs{0c7x<2m`9WS<#8z)!9u66mh6Q_E!h#Bd{f8X!j2cN_ZP4UeYkTG|9Z?Y7IBz3|78@A$bPCO7Dg"
        "g%7YpnS#^zL`ZviI^|Re*f`h(7@Iy#e5c48nJxF7H$IMgZ+u!)YI@&NR9qafdDXCSVjHlYf8Prb{txf7EMS!Bv)`Aq`M+_=bp!{C"
        "7F9-2+wt8ydQo{9?r)Pxl5O(4Sk0&!bo3@z=a;32vrqrH9lyt6<Dbb1na!r(9r?tC?qBsq8VK=e7%mHmn;O%m#+4p<r`(iomHp^-"
        "M>J|7T?=M>o794Ks78+e?i}ejpL?$JTwnI$P8_eRNStoeA9sJ8ADH9wde7(RC**pC|8JMauq5r(s$r32-9U-oh`DcJO>3s_kB?Rl"
        "A17C5r#)>)<H4A;E3e(f;l0|a?yN5qqxa^Y(q?8BsiBMkQr7PgIvnXFdmS?(l(*TZ@=LuvxSUm;ULZ%+aX&$qtGYIm?))^-&GyZU"
        "QEB725&*3n1Sk$cK;=n6fENTo;NNK5iq{2i-~2C7O9KQH000080000X06gx%4%Q9;0826e03-ka0CRF<FJftCZ)0I>XJu_|VJ~E1"
        "bYU-VW@c<@Ze?F{W^^uadA%B4bKAJ_Ju~?aC?7hcBhhwxAFXn_hf92IX0F#vF71n=;Xov0@yxFfq-~u?|Gm2lfCMOtQSLhRgGB<1"
        "#qMIiLC^EvwY6XsPiv9Pw`nGrNYW-PZdp}}ir2hJ%Yw-rhk3TyvrXCTm~42F@T@EZ`{l<EtSB3?Da$(<dY<?E+4E;<UX^vjl61Sx"
        "(oOfi<8o)e|0>I(d#M3snfI^a`Lk_Z<}5C=OvFfqWP0v(*%pncH9*B1#A+sypLeB9AK$?^67&37BDU-{ZK6gzHc^%qLi+d^ES^7O"
        "@CQWyJ#Ankz$)2KKe9w*X`aFkl9ffaf5l$EzucB}z7-<5{CAPYWg?hV(kTH5#Ia{-0T_U+qES*n%vCC4A=#4QcA>1YwDG-<g%_yl"
        "S~P84uoaTTUND?tun(hmz^9h47W3=%8s`0#9-#Mh%lcO)GAY<<9rO(AV~lDMmv!RT{9yqH5;rq;F=Mi9>sTy+@;YGi?`c9y1MmWD"
        "cq@QZwKi(@mGA<#TrwD`K(@g3J_5PNyXgMf(+e$*EU05LH1it)tcn@qTM#sesN1y6l1NKZdcmxpD8z#`p{Xr^g_hLA#$gQu;lPq&"
        "m{>rHDjwf$8&8R?TP54gub^rj?k#FIOC@6CMJ#*^?M%rIPK#jri4dOB$`5R*@D#TB56@b0Q`cqfdwD7)O2wh`)UHNS&W8Oo3DBKY"
        "EAUM}0pA2RNtDQ|W*?Px)E4oM7q<e>ArXc!w6ozkp~E;$IIidkbIwsMWPe%`+DWq7Lber}HqcXxg6AUQNfLpPil*B<=0ypH$upqP"
        "@c=|bPodxM)GY~98S#xQv$hd_&;tO?UKo00*&QqxeIs3%j~@x9B7b8~EXK!sWL!pQD35+82?vlDdmr}$mBL*gQeejmL2T$GaAJhC"
        ";0ObpWeqG!(0O2w*5Z&MO)r?%SNjZbkr6vNQxJKU%El6{cKl&mq}vph;JY=Ae*=H4^fZ`F0+9+2VK4@uHZ8XvzkEXC4&DHyDK)_6"
        "1OTE3)cF!!ur+67(-}aF7O?e@A2y!@dCcs|3~s=nz->=X^ji{Es9r2Jz?ObPQ6&11F-9oo)W?R8AKA|(1LL6~-Y@`X<(46D-KH4G"
        "BaE{^rw+Fv1n-w$f8%Qz6S6B0ev~+2SG00C<7PYms}}^}PCO>*t$-6X##QDJ4gn*Wvfr7p8uiGwyKRr%@(F52JmwjcqzO2G)T#Oy"
        "V=rA_d;1rL@)yc^y|Bdd;)uqh{q^A=Q=y~WQ@^8Ud|s3$-ohf&QYP=VF53!b6;bv~M;6R>&$9t~n^xRoFy!KE6spD<Rtf+UI-*!p"
        "qzf>kg30*K?xyC{Y+!Ixpc*POQ2WOY0qR^}Mm^Y5TARkCL)nD4aJ5ejlJa?;LqwLwqL5;^zznM;ZE;tW4@I}~&?inR?-hg0MIt8i"
        "Y~@gtpG1@|F0ZfPFL3fGHN(DRFDL!$&yZQ7NoRt~hBXf*u4Q_+(5NDhV^M(%hP{P=3<NBnlGG(+emDjSIy1?!(N*+9X_1IWA{2d*"
        "DB4^=<|=HCZ1)e8(+c+evS(r8ddy8N_+3BL)cXMdq|Z4JcAWkh!7bH~6|cl2vS-PKamfTe3~+0{S9rcOAFR=rUb|43DyBs%T*iQ`"
        "m1S>hln6@m#1>xCLe?IGv?66Rk+PKYHblfBBrE>WwieswR6|}>;L-`!bu|6CW1jA7+N@lkDvM)<wUreVY<c}HU=a7ww&rBQ%d7Bu"
        "24~09LTm6#P+$eqy<7T>>D?9RvW){?uhF9NwD2uV7Lou#Cbj=<anG|fQEXgFo>v(LK3Lo9J~>ZRbxB!>|D?EEX|K8_MOR~Zq?-3b"
        "pZR5?@D-BacbH=W2~JwXjfp6l9Nd(%oyd}T**1!Sj$~EQhVSXe#}f!3R<)2C2}6mAuVAvk8p_ZnbI#y+a{lPU4-A2B*s~YTIRt@5"
        "hJ(GZz$d7e)q-|0ibqF<j-ytBvt&k|tCy=PZ!EOy)xtXHb+_wT*oK%oEm=ESrpVLP48TmD)(zGzYRM(i;hqAQQb4ARh(Kuu`hxV("
        "!UOv=8=VD&2iF=2p7i^;*v>g*yqGBr+^Xk!KeTC<u(~Z+E*hTjhBMv1=vyy!`5=jF-lX@!Dx<+SV)iXV+z=7I@29G=%s|Xk7lR`3"
        "EZalm!%R;Wypp@Jxm4+<%1S^~;QV)xK{Zh1!9USwj)o5KWcQ*-As-TyWeFKV?pfzxrN2Sy3G>xnw?+zaFVOHFYVcdg(Ql&Akq0n="
        "^NGtHwq@SFA<pP*Kmz#&JlHO+!cIQw!3dCIS|7o5=+aWPwV`}aHL2s}e}4nR!g)pyql*?m;+>!#kZ*bU*+}9R3^vmg>($67xC{JU"
        "!O$AZLLi5pM;4CTgGIR!C+!{Osmzh*1!n(%(&EAZnjqxv^>n=T#l^@V{sG=aNAoy{Ev|70{BsU(HUXy+2{@hmRAQPs39C&aQbofg"
        "EcKe8_>@97AWuOD<HOdmmJi1uhY@W52dUbX$`tp?6HY-`ivF4X91EC6*-QiI3S#=$sFIihQN_a)gsB$JKk?yl9*)`wI(`^av(mSQ"
        "Z!!^W!N-WYs=<}}_h`??SLlT&>s{=lhe?}PQmva&k`#e5$mNHc-n(Y?()VWQm=@St1r}ROX=R+I%O80r#d$@rA$;0L_Ob7E2^-~t"
        "LIzVs;(S{RG1n%hVk%xb=s2fBBg7`!!jxon-&uE@d}(@#9^vt(yPPN@y4BFMMlmlSJBS39O_8C*>T34=VXb^vafVs>fK?Wno81Wb"
        ")Ydg%x~zqqf3nrbE842gE@3kWl!>dL-X^_8sbB6|&<>;!8`KU24Q8#pM1g$bX6`V_>jn;Um-G5A(mryURlB>bPiEG2y?lG4u{Qf`"
        "1VCdMpLDtf2EuJ7Lo(xA4dqAOO~w%UXc`dY_#VmaO>hvL*O0d~+K31267qXFrVt|$f;}AqHF=8-@>YrjZy+SGLnby!hqb>!UX|dC"
        "_7R?&mKd6jA;$zXyEA_j<WON9xz-}wwpm8mfM0v7tNGvfe7k=7=BoohoAqBTvTkFQ>h$?rmmE3KrwcN@)9!pTv#f0x;0Mi$!L5T4"
        "1131`wTQ{8hWXw(tW|XhDKTy^5G84s)1kMz!`}IAbzAcs!}xq-x6^^HwX0O#`0~+C>L2OQjv+j5ie6k`yXUwqIilp=Xq(bl@JOH1"
        "^IXLL+y3bvZ6U@j_ScdKOS4FeiSAmoHmIRvk!w++8d38$EFl>Or?7!7xdiAU$!!0j?s9+-G%mcb_A{Pu628!&VJ&##Uthiaid``L"
        "0fX3hM>U8dCv2+(yMlz!JyZ=ange7dhY)Hp>?cr2#(26`2JnxvpjL@05E(k3IRdMK@D+n1GTlO%3%`4+990>l`Z@aI!Z_qU3mCos"
        "*Vq0;eo2mz?g4wMjw(&cpO6pf`?Tbed0jz-lMwDvl5`j4lMMhF%kG|{^9S7`QWyMWfHyib*`(Vt=Gm>tbMA<1c!NIaB%Fl?zbUu7"
        ">=9c@E+ze(<-JDI&ZoNtyH_521`lbUt%lf^ls;Xi-w{VP(gQJRjZAXnaM6{h*ul81(`Mg)9?^SWRF$PwzNw(-ITUvH0(GO}aL1{g"
        "ySPZLHZ16yLA5ey)`K{&0+IFuh6t+|K)DnGQdQY>P$11naO^ZdkWvDNe{kwL;W;-SAZZy&407H=DhaXUQ;Xf3PIA~sT7n343oOeZ"
        "-kN@(6_!;uh61z7oFUjI;@)u(c;SvK2^0bh8ARqe5qQksX+y&0q#V6$sWGHQOVgi?$FAC=_&Q~wUIt<5k&yp%2vCmn$Z|)WD)EHM"
        "jy)*LHQ=>t;N0q}rQP+d4w7p}G?bhWHdwf-S55|G^%)6f?FvfQCvbG$dPN8D`P9V`mAV%&eLq`KY2Wgp+j+?U7H5_6_n4F`Q<|XG"
        "f4%@alTt*1pPuM_#;6&J4oKIE{pH*Bw>o3Ov22EJZ;m$~CGZcl9Es%9#dJArIGRme?{K0+NS7<f>(F_eL@zR{{u<fYO13rIcY#%!"
        "QJ-q0OGR;IqCe%-lP5YaCObSetrEmjvrcAx3TV2=bWXc*AkML#@8_bTu0Y({lNN$+*UkVO6dA9RtKs}?q{-B4fxVBqW%CU<-4&`N"
        "zwR&ctXC_gdoS=~IZD#OdM~?tDuGO{8Rl&0FVKzRV-Dyt_8`*Ropqz}1BWV_J}1gsiQkIi1xN|`9cm4Bf+Pka(=b9cQ#Tg6pTst8"
        "0zdjb;Wd80hI_$6DGZI0=|CwV6U7f{_NKFx%;?G+Ti@UgU>u`HLRTxp8w#)t{Hm%y5VUmNuQPQ%=3u(!yk_4m8O`Fu^<S>9zJ2Ms"
        "VKs?XF)8?SvITRIifqDBXbxCY-icy(p(-9LI6eSbI{hn61C%~4HN$F469;e(nNA(2Pt<~Sa6;gA*24SO*Dw2O3V(RqwDeKm=`2q-"
        "laNPgk0^evd{^UwFlBh35~gdi-n*e7=_5B$FBm3!sOmVTr|`o|y6z)2vz3W?F5WGekvXd;L)PJysgs@*&4<uhHZ0BJ##d0mn6aOB"
        "`Br;&C*nK2*{<-DN?mmUL2(J?7{3*-7_Xaj3%lssK(+yE0owv~@B1QFb5>Pi>W4n~R^#t(-v9IMhYvSzqSya=a})ja_Vvxrzuf$S"
        "Zp?s&>L~&r(PqN8waF@=LcBf=$&sN4Ha+SS_|5iMC{kMd0Muhhr+$RW!PJnxt2v{&r{GcFR_j3PDg=atu1uFBNOb=dw7okE1Fw*n"
        "Q`NGE+|ztC(h*7D)ZiC9dZBw~uGqt@UYPr5k#rxH(hPn+;LOrAj&>to+MP7a7+tZuenWak+I9fp^j73ibK;Ar&|!0!>FQ~d?XLRW"
        "h6%M>ESf(JKI{VNPa{sM-4M<3x7!Y4-L2J`)XC-3z1uO2)c^fmImv{mu^-J7aCeQWJO2w%O9KQH000080000X0KrM{K}!b!0QM6A"
        "03ZMW0CRF<FJftCZ)0I>XJu_|VJ~E1bYU-WVRU0?WpgfYdCgc`Z`(K)eh-lUz~SzL)Yx*`!3+kT(YAAGivn#EG}#3faaN@zD&|BY"
        "1(M1Illkv+&Y@(>vXge-#w}uzJUVxN{AkYknYSwU?BC~ara!Q^%CkoJe8oi5R4Nx<8O`9oa;6eq7R<`VIOUDKXHJX8t&DdG=lt;C"
        "@Icj#v7Q;XezbD^=-OptO)lML)yujmRRuDoH8mPuRcgt))l(Qw*lBC!DOkM2AsQjkr7BG>sw-L7BFT;RR^;CGZ+=?)_y)932II5N"
        "WmI%{P{@)wQOXR1%FujtBdS);ne#Sg(|_Uj{P2Lm2Ul6T?M{%6gMcbo6rS*-jc2sFnKG7?B+6xC3F&OTjN;v8;WG60E{fSV=^h<C"
        "NjOtF2sgrdH(I_@{z-wwf|mDao<Fi6^nH}Cgay}KBXV#XXN@grHfBGuZ(pzyrf|4c(Zv?eH@d7)PO|^t>T7F_jd-hF+hEaTk!@Ow"
        "YQZ}NR{GYi-RkQ`T}pB;qUTM~eA`s=0!9;-B*|rGAA0mGn1a5$^Q<<7D!UGMGf@;OJQcE%^z<WG{$1wS#9GwS&6Y|Fdq1m`zV2GS"
        "t2-83Ii;N!8nQ#~OoBU*6kMgL1ZGqdJ)IEP5}xAE*9ia%Rwhmgo392fX8Hf3x9`qba4VR2_$3T*S2^|&JZBHNY3|vBQ?;suRsJ4o"
        "RLHxB<QXJEvgrAXWB5Eg{~o{Mm`^rFLM#KLGWjt1e+wIGWg2NI+kIRsDGF&F>?+V&f=i;RB8Up=QnKk3`k0AXv^YD8*)n|plTSL2"
        "$dP!f0-d@iy{e&umA=YAcZO!N@KWy)wG;SraT&(g)21pWOv_vB>s=FI8w7Btg2-V0Yq}k4!{e=7uUrDaD|Hc#a0!gW@y4mf%2M5d"
        "UszsnY&(I`bUJ-l9O1v1Magd7wgVs~Y05bC{>@CcRmDe<Ai^p=L<M=*z}A5C0A3R_j%tEWqzLSDCM$@ZFK2w)7*q-k!e&5dx>XQd"
        "uZ*_KC~%a$crm|>!!8T3=DwKe&~l+(q$q>Z*P~rkNgX-qBT^X?-r`gkqAxi5qe*h%WC;}j+@s{L-S~6#&Xc`YwBbaRnXwtF%yLsz"
        "s=$8Spc&t8>dnT7y1shFri<bzdO4rKSNxJeg@?meuO6@S<9ne?RjaQ@h)X!yMYaK6+&WU<C~I^rwT}pL3!o4Iap6yoxF7}X@yGR?"
        "-B8}HC-4BNGcM^ABnl{i0~KAg*Tb&h0B#wN|6D6e+NOZ?A?Rhlo_u`s?rrw*<m|s^=dVAc{5ga^Sa|#X<d>h{r(g<t8e2N7p1=O|"
        ";oZld-)E<vPClHTr~G&nGnm<giX5oJ;>??@=Q~mVFyF&0x&mxYqg{~=qu4&d#vnA>31xy`p#@9KWA8>nXn*-PZ0p{`t<J{cuBl&g"
        "E|Igrnf05p00NMvC;&;yJCZl0<V<<Ygd^w}x?`YgQlg546meoDP^+6QW<?1DTe~?M!AKz;z+B+GDT_$?j9v|?-=}=AR|l^n$+Hp|"
        "v$~PcI;1KdS=*n^t(0Zb<)Ya-Bu0KB!Gm3x)YpZwQ5d-N+_pe_<(+aqGuJeHLd7jm0+}uEy>t-;i*S!hrOtsLz_I1a=?^>}sK^Zv"
        "Z;j}G!I5uH0P}PFn;QTg(R!2Hs!U4(X8=yqP9mx4?Fy_BjZC$<6;|`h?Na3{nO|eAq7cR#R5DH;RHCi0^qXr%Vp|>Urnq<Yh(#A*"
        "p0x8ztXVKb$J7O!FGmo|lwVGERVdxGJ`Eghl#@Nu`E8M6z4t5ux}-b!xle^?0F7WrwH_@~9(JX?swGII-yV+^ciS_hD+wTn0enp_"
        "?uy{2J6Z!amIC^vm@^L9>ufj=N3I3f9p(ut4?tEy(Rq9^|L*wsa##6q#}a+r@GzR|E=aVCi9*q?O6*l-mJxpj{W;%>yT5R``x<F`"
        "^mThMM3`HNH{Sl;@g^K-M|bzC?tp>h*4Xj2*zkmJfm1&Q59<u}Tynqe?OXCALip+2A9*~us7`tjQFJg8kLth2L;UQS9!&j%?H&%y"
        "+-k@CN;!ZxL6<X1Yr%jxTuJ{UBpg<}v|3c%Q3{|4VzmLOWQ&Uuz<;y<`1m5j^u2Exs>$H$tFL4+jI^T^y98iOAQC*?C!0fQ8II4o"
        "R=^_y{h~s~1O5fU;g20ztKs?zjeysf2ye9eezkyG>WQxInaFe5z;z1ZxMg&|!kdkjh>l>zOPZX8pMFZfIKW-%Tx%d~rxi4kA6$Cr"
        "2A$GZcdO+KyU+vj+AVF^+~?1q`7o<Q#=E;8@G}|$$ptYWwCK%4AVn`1?%6^|$+MU7LZgxi=_ZDI7H`!vsD#$|2%&wkfJ)y8;3*xy"
        "qo9Y?>p~=*$(1#26FuMN<~tz{F0TAbx8wfK(^G#;^xzswsUa8oT*5}~%wV3wu|)z$3}>F^lwB**)ZxKjP)h>@6aWAK2mk;8ApnaC"
        "H|N^}0027+001Wd0047xV=rQ9W^ZF*Y-eR{ZDB8DVRT_HaCvlSZ*E_4bzy92ba^gtd5u<4Z`(Eyeh-lUK=4Ukm8^M9;fJ7GhX6xb"
        "%)z$8FbGDv*jy!2BdOSGk^jCUMM|_RxATid^6q%|-FJ7Sl-{i#uavAhV!7UeB`>@ZCAUIJvKMwocv}j~pDM_SDO3X`S#{F#!(UuW"
        "A-ALyd@Gf)qA=H#(#wm>3sE;pTf&VUzW2SfamTCA507_$vtRD-etY=1dc1$XVxK<#e*gY6SrUp$%=imf)L|M5rAsK8R2CQiKw)w5"
        "a^o+%yeQ$B2x%JJW=&^zO0qtdtWYJo{Zme!MFo=Au(-XvAoxQ^AK(Qvfddq+1)?<3u-5`VjcRoP??|J;fPTRZ?+yX2l{SRSk|?`F"
        "-RNH*wlmvsZGd?}jO9T|==VL4gLuJhu^Xbx`skLAx`hQ|Mhf28Rzn$F_4-lOAe<FoG!}G5d9~-Aq4{w}7NefkU1G)k5Z-I=Wxyr^"
        "vm6i_<>633V@cLq|J6m6lRtUYLiqjzvC{^b;@94JAN%CaV^sWqFOX0N57wK0_gM|lVp`|m5SVXc+NYx8#$-X(!3tZYyD1zKZ)z}x"
        "ZvihfRu9TycX2G6a+@~|$TB+?aKCS-Tb^RkA9_osE?zU{ux2bjqx(h367{H%z8o1CgaMd=e_N0R(0Mh)C2EjfgA5uD$`ItdGE@c#"
        "^~|0WAU)@0u!oL@uXBQtTnQ9$>zqm@WGe2KQq2gz9&e@e#egQcAL__`FgP@oD1@EK>4X;fNe1)NKqkJ<`h^)vt3eooEzLKI#8b@$"
        "reZ9#Y@k8k)?$<EvqTr<S*tp=+ndsWbt6*_kL?ob#&(l5Gt<^J*IgDXCKyQgP5mX&55rtv8yXpA>ur828?9rzmG+#_Q^x}3zrCi%"
        "yl^e(|K2!AfB()wZ06S25u-x2@?;ie0i-;|)sL8Hb@_oK@sdIF$%ja+vUq@HSX6BZq(UO;t2oTC*K*g{%{#)&T39<OaP6uXaYBW0"
        "(jbu>a}p;~0{O&+k)&~49Q<Sr4pOHa7oqI?Gz8lr#C1@8L+Ba!H{{Kd+?-XKWWh^gtz?PQp)6b$+z6O45_6Oc2OQG1YiIMpds}NK"
        "v+mpIW`maf?zMXpq$fFfyTDs+#-#Tih_Aq69)>XbIUVP<NizIFNM2nPYUf@$o;FB{n6t4u(a6uvs&OMTruPXjB*4`d@Ut#?O$0RD"
        "%=~V-j>u@WcfbS>6HtD@Wz_T7(9mEeg80H%Zx#u~{afX1lZaf#x^n4`c;z!9J)x%8-3~<fQ3KedMo+5#)XJ2SsruLrPW1*Bj&0&U"
        "P)h>@6aWAK2mk;8App;T!S$LR006>t001EX0047xV=rQ9W^ZF*Y-eR{ZDB8DVRT_Ha%F5~VRL13E^vA6J!_8}H<I52<UcTN;4C?s"
        "9^1ROmrS4ooY+o`O&r7a?g2X)G<qaT>P)M<(~lX;llkvg50Xu?*^(ySJ=_7O%_dSeSu7T-idDrbP83Bi>++*4#4>GDQOjJWjcmlK"
        "tVJrSI=fC=DSmiKuN&Dyi>jQ7MpkK!Z&r1=729$tb8#cH^`>p&D2hJ+?DNmEZB^E-Nb5B;ZKQdyNt;ccU7Fv&Hf3Rc)P1{V)3sS{"
        "eztO3t+Kp-|1nd)357+O=W@|zWzh)L;JdObT3M?GRoddP`sEw=iCEIR%WPFH(tItq+cd_B#^q|2XN63fRjYgCWx7lj9nN~0RAre1"
        "725PNm&qzC*0Kg#6)l2%`SaVKp1l`y5!rAPMZiRnIJtfTWP1JVrx(z&mhqz8Rxq7O9i68qA20A<k59f%PA+aA&;H}<yJ-6PXK!Ep"
        "{nfiyKffMyfQISG>f-j#kEtF0Zz)$|nXP5hPH^Vav(G;h@VUv_4b7dO#$_doNmO4(Q;{|>pIW9{)f7LqY`eP9OBN%&69tvq&P3E~"
        "(x?CamuMPq<OlW6!r`XQTAAP)(}**1o#vg?=vbI=!&ZK1C)he(cH63%Q0tk<ilr>t`O}$@Mbp(XNt;EM&0nT@Bd6j^5&cp`_-;`y"
        "VJ+rS*RD>!vPn^wH_c%}3;+a)vdd(?WX5Myku#bJ^2?Ou2EPHys;9_S^dyF#Sv8qjSf%x1lU*YN<+7R7W!cU|k#1!(6J^&{U8}JL"
        "k$NtHm$zAwHEp&K|9JIAlmMU*@<Wwpiwp!U&&!(}UVkHsQY_NN2H_~76o6LU6~M%6C7=j4E3Sd8WxWHAy1laqqhu@oM^?Q=9%F1^"
        "v2TFP=H;t5$@3R4e|+}-#q*g!cC2O7<nmhP^S==3)ch>kA$f3yL>`bTxt!31r&b&MG-X#WBr-Y>MVwL}w*^qOfLGiwhLOwsT27{M"
        ")}Vx?Ez-3uiKzWse;DHNG;5^zJ2Kjfx-M%F2PL5lJ3$6PUK$@A#zcxtrby=gS+m>1!e349H|jQP8sNdCtP|`F$_8PdMZo4r1s@T;"
        "gy;D30vA$C<e63gjL{U9?{e4523{x}L4Sd;!hez``zY^x$g2{-S*-91p3rj5a5n(*W?MO*Jo);o$20MS{&~#*Ob1A=%2+c<((Wp0"
        "1Gd=6?MAGqP<0hm27(H*u%2la<M%iYF)1|yrd^lNC_sJQTL7hMhd7j%zfO9Py5CIXvRJp9`IE;_|12H}{EwQL(u=n}<cPTBOv!Cb"
        "RI=XsOQOWvZlc#lL2%c`Zn%}(OIc%6v-s-uov(MVGkK%*ZY>vOy#$T$XeL_VzJg|0)n!}4-Z_6=7O<?%rfeJfZfTw$OO)!S5mk2y"
        "s)lr_sPeAqbq|gsYq3Jj*nA@-sBN(>K}*6cc8#$XEX~7vYPCy&`EPKwsx<T_hK|2=V0K`O-*9r+1`zJHu+h)OxkZ<(fYAyN*lZcu"
        "^s38TIuDqooJZKV?#O;{o-fJ*w<PBo&<w~<Bcb|t-T|)700>dl2SH>X0stG~DrtZ*UHm|BB(i!kn2cOzZE^)0q&om|qr+ViRIl4J"
        "UBQmxje<PlAa$WuWo4Mg5+}6|aaw`&Ehm6{y#@;h245JB*}y)uO<}33H0JM<bEQP^3G9i*8Wgj$Gbp$h3dAMC!^OU^wy5g2_zm;M"
        "e<7>aip_b5emjD-m`5L*cF8({#G>{#B;-ICRY{1aE};DuwdzobI1G1&P|lr47qdVgT0hm8z6V<=4n0E?VuE4-WStnk73Rq$SS*`u"
        "T3^A#D4I^EAz(IdpY?qmmdD<qzJ{Nzjy;c$^<su+?CJXy8Hm1#tf8F~b&QZ;z5$F72v<VRgk=-zSGd{U-p$C4NUr3rnZK{Wv<yMB"
        "q^h6Du$|u)Cb$KzO3aLL8j<y@Al~6=;HI6kEthDuF&rX@o-5r4F&tF}PAgQxU24!qqH7UW7^mHbEYH$<m)N+YLV=$lxIyH51!VDv"
        "rIp5*L~WR+3#+WL)=Gb!*}Tsh4SU%7X&+ay%Cmqd9e5uYwvu%C5`g;581z>{KsCcysF|&2lz`qo<RBAfihvf=AHbQ_5`4KBZ-)93"
        "S|IC>jA8Io=5sBtv&P#JgYC703?>CoZIrQ1d+^odA8MRj%Q{<Si`3t#gFP)Wqvd+A{TetQcAst59&Fl33?<a%x=yzVI@w1|0|5{A"
        "`(WFmthZ^NeWYn7RSC1-9iA=jwF$*Oj_lD@?B{|ux;c!0T9gG)K2I=?)EqhA!&sNFQ#4?V9&Y+u2Q+DS$r|j@qZ_aCbbSo6B(cX5"
        "u3P~RW<UzPq3!E29e)LSOGzZGK)Bi62;`x58foaMwZmJ3?@R{mN9!$aPiShL8@+cCXh@CSR$!a$QADfg!Gj0lRd4G2Qrx1FduUld"
        "4=?V1f#-qE+J%CVkTb#EakHX+{FC=0crXGauzEZV$6~&qc*l+Z)a^^gu(r4n$S^ViZtH}aO~rgJ!0!xDk_}-Ub;B?w7|`*1{1Rhm"
        "E2(xEOJX?Mte*3KreuL8>$>bJ@=oV4MqrAJ>LLeYWn#Ss4C#%B)xQEG?G0acS-#Yf*GW^!MZ)gFAk6l2k?+WP$Ym=jTA5+2V<}L@"
        "G-81fH;gI^6D$7){Fu50Ii}#6NNY*VOkrf>@8XcKx>mT5Vvh!sX$&fK$>GmI6jITSaRn&URbg)2GZi+FAAm-U6LR8E6LRkhda2?~"
        "-)+hgo~NRa;Qnm49noCRN{_b8z}~?SL#!RT2jX3uuBCD6fnH4sU{)!Nu*8@E`tLv}q(WYnS0c}@a2yIBG#hB6#)$2Cp{Ea&jki}Y"
        "<b;11BMR|O$}2YTEhkVP3dHXvh7G`!eXjAH0GPQ0njFAZH?#=U(0mkKO#_Q#?ZbsWQp#_NRkm&{=M+C^B*B0+DP>$^)K><?K(mqC"
        "JLewf5&rUpUMWaCEr?fLo^MeG$cv_sqvbd2L5>PJW(Dy#h8mY0hH@~f*8}l`o)n8!WKzf)_J}UL#^w$5U;>{ChYG={ZqIH*KJW&>"
        "m=hAd9|#7(hf$!N0yor^Y93@zU7)&*>?58i6gxyS1C5<#Cbnsjt-u=uGzP~*6nYM5#k51;h@mXA=GQVS+CnzXqz4@8#c|Fi9E!kC"
        "#Njz`1KeYU@okL`9L8V6K7?C#j7lp<H)JWV`yYDh^u*^^`}5@?N8ub?sx3H-Y^Q<_S;IEK<Lm*8A6UPMIv^N_F*Q1IKQMCn&i9NM"
        "a>De-VM1K}_+wU$*>eFz?uL&Rczj9gO|BVk>U?&leK2*F#yUo`PK+i2%ryWIqXl>>Nvj$#RPqc=Z5&W9EiMcy^XLLtp}dl)y3QkR"
        "7Frs1_*f<vW6=wBDmx)VM@6(QXiBtud$XY#!gTC79kmP4Fz`q@d`F_RPg0CiZu^}5^ZrrP6<0-hQ$+qDi2XS<-TggnIvmYmfkz07"
        "1~;bzVO|<B#H}L+*2SlSbr>GV>cD+8wT>RN3lLK0_))v5wr=ko!-yC_LR;vW@D5=Ht<gkdB?*ns_iKmSh~3LGEk838!CKkd+UYHh"
        "dUuKxIc5E+DKX}R1%^;&rwXzj90wKiF{6ZH&G-o@i@@s0>ud?Osob{(p&xFgLjcyjGO?eJVZwSQQ99;K>7=p7D#FB?(j04!Hn)YB"
        "f%Dg~_N%TSVuOpW0|6X4k{IjU=PnN-x<i$^C%_wkFgp~j9nk~h_?g{u96P5~2$aV+G9MUJpige2b+i47PAIAxeFm_PXeBT}9o>b)"
        "0YkHlfsCl4VHQsx{f?=plf^?%^kJg^VKZu`bDxGr9`#n)8d<T(JD3X|c8+Pa9w;d$au&e!2FIkir#XeASTHu(5(JXQ-fu4*mR$T8"
        "96m{c36hCu*}cufy<PR$QQwcB+(UPvB{#w->IfTF5<>};6{HPF<e0=+ma!dNXT^;|Q4LPf^*aR2;w~#XFis#d5(^U>yeziNJ9|S3"
        "N1F>hu`Grgl4v%L#|SSvK);;pETHiT-?F7xq!ls(X+32yDmzH6DIU$atyD-fz#)4k?aA2adVZRy4T)y4{66s*295XcMAXTUg`@k6"
        "ja*!zjOplBua6VmbRP@iht;2_EAQ_+1=dC}NVcQ}7@n#Rmp-r*=oyAb#5f9f3%|$wJ#e)e*#CM=rc?37oSyTu5#E8m7WQ8pMxV(R"
        "z<<o<9>#E3CNVt{?UQZRZ0+m|k0xoc8zGDFWXRm0=@RG>!veE|M`7XT(1`?o?s1k#Qpxe~L&qvk@cdBYv}kVJB$!~cTCQY`VtsgY"
        "+&@vA{=i|*$m0ROe=@reHM1ho%!(QvgMc@6S*+RM{HKPXY-N$C1Xnj=La!)hGHS9S(j$<2o$bLT(`XE8K=)WKZ5Rvf=+ZhHpKAnV"
        "r5(dBjW4`Udz<t;y{#Pb8jd+iyxp^jfJ7|Qs@0A`RB$5gGFz<@4yGs60Ayy&(HvVq&MkB%qD;F`k6?5F-MpEOW8KmkL}#{IzMG9|"
        "t^Xsj1)S1Rp#_`&Ml2ls{Bf6m#EKrOl&)yHsv<Ma6gQCL<^iOV5|&=nWm=t>pjTip)#jY(f#;(^iePTE*VKh&_387Zhr{*Rd3k7Y"
        "BbDXkIH%!k9u<&KEE{ODl+B{fDkW$7G{t%t1gvS+>%ky!Eo2B;eH}lZS7%G(XGeieWLtr)q1M`7<_F?+*=866#$!<VAzidW=Z6SN"
        "aGr@mUdvi^4g70SOEO$Q^`J#@vq8z`%7wTGN|~VqRv)La^5AC;h|Bqf&phf#jho#2OzmEWXFLY$1^)~T69o%WruHeAMjQjCgeb;K"
        "+YjxY?(XSm=sZ+^ywvuICIS0RGaIPYlMkiv9D4<8=AUv7&NEf=9Cq4E|4%>C^bot(%D8=lD`e_S>_bf^mDkd8&P~n9p%eQ9e4yy("
        "umQX9{{lW>q{r4jzyo^~rP`lt%<QTgn&LE0o1`k6?89UvOW3W6urUj!fT{usd(hD;3H(?btg{*hvx!!iC5L-2m+g5VBCc#mIL!pZ"
        "zYS~br+!1Vm}fp2{dTZd0ua!9O`jQgW(ig3y-$uv86linz|!S6v5(r%Z0+thG&$tS0&JGNTs!Ct-;%fwM#F>evNkyB-vhKT++bSK"
        "k3kcP`KKE5fbA4Lm*6<drJzwzsLB=XB^f5_nEgiOlj4>Gil(fyHE>NXFuxlXsoj)yv&pJBnB@Bo5m*SSTs<XMQgGBnEK33>srEB#"
        "HljZzz}na}tA#o&2t$5diiJexQ2D2bdsR!5d5mXrz-k2!k4bLnR%?!0Zc9*XVX$vhOp>Z@vcBYI3EM8<d{dz65}cXH1W6aL2Q?=+"
        "2G~2$r_5QVeE9YP4Moa}3bTIms$wUqt)+0Rx=^DJby^VDpNO$ei?m*HC(~x8&3ik|%r0(pXH@N1R%d}4y2>t}tM*sU;dWVB6^=AO"
        "T(9%;auPin><+LKR(%O!kk%n;nfYmYnFkgOHMB$>;iinK5;jG^K2>gSOfVSrv$E&D1(TRruq{Qqk)pwUO$BH<Nm6KNB89djRDuNz"
        "JhNO0MV~k~QlG8TKlQ0jpH^%_otRF&7X%F|O|>|VzEc6xwlr|YKv4Q}hSW})wp;Erm_7oHDT`d&&M;`IAdhqLSOEn#K?M)OGE;I}"
        "U{&N~Z~}!@Qj3;%dCLL{M7$}Dw$NZbYYB;#6`d8PRc4xMK8UjS4ROE6Ok1lQciNZL4x9;We!oiGkw}#txpsn@hKuF_l5<VBgO%s-"
        "k1)(c|DpiskSt0oc&02Sb6}u}_iwN5KLHXg2ElO?=I%>F9VL!j(%8XO$}54N9^?_>FyvzXJ+~p*_OoXFFX!TkqkYs1t0Ml0<hWkl"
        "yp^m!fUyg(ZeRp8#SG+HsT<Bm<tqd*@&oU2%f)~y(v|KukXB7RCb?Vg&mYK<0jmUUbmUCg+QM<L)=>3~y9zb%gihnB@S798NMY!d"
        "uRbNoFR4KLY`PyX=e~EfJP4g<ibM$lwnO-)SVLE9!QQwA+oj_wEnSX+>5*V1alK2()y`ICcmO>X+SXG_mrHDh&J#^WmoQm~BKGGr"
        "5j0O!M}Ez<6OBrB1oP$J)bFLdcJg$`>-DK;0_~}42DaOz1UDg1EBpQl%W~m8bYQrsvD7xQOSie+IJUA)>2emwj*0TLSa&cS+Q^Zl"
        "L?JqNu+24-mv$zi682-a+*g-FDR075@|=JNa>5*XV$x+o9r1nwic)Ucwrb8!PdD8fjO;ZyK6pRLskXpRZ?&=bP&oz<FYZq5{qzKz"
        "o?e25bh-s{p8ojiyBDwDy?7UI-MVHf&crn3HQ4hUTaC~VI9ba=vNPZkK#`YRAo}K(VavANnfZUq2z`$T36;uAUW7(L=>%M7zxe#*"
        "<iti4*Dr3;l}xZ4Z=#pbt`Fp5Kz%F7Y#dhfKpe!3+BrYoXyakU*ZUX;1I?~JJ4P5sH5T6i`<(J!KBRes6-GBpyw2K9cZs(-!1C5z"
        "c?H(Kvy|f~{SG5&H&pyC-eR=xch0_S#P42HAcAoI?Ny}fdhr=oXlF(5kT|*a0R>I#rw;;FFcycAVW3;8%6W+KU*J%DV3kzzI}YhS"
        "yqLmky^(%0F2KQv%Q?^sr0LNkM(EL_GmViI8OACjt=*j7Jy6eQ9@Mc$f1|<=47-URx=X=|TfF<>8F2?Cx$X2EPOWWHZ>s1_0xu0*"
        "#pu1}GVO9UD&E^jBm>_JVbR3CE25BcY27K|o#8}-RI6A#%k$phZ8P}APA%G6=z$;?(`e~sNouDVV*Gc$x<Xx3aQ&1F222^&W!~ym"
        "jQ*|+qEW6d#jwhO;{ndE$@<0HdKz@{8(G7z20F<46muojumJqk^M>G(iy4dW-~IH}U(JZk05F)3unbRTe|qw@z@s38Et)4S+CUhP"
        "$amkM7q&A+Nl#~g`O{aTD>7(Kfsg<}%n|}oP0pd<oWm311(kEC!U&>wg>(P}CRlY%fE>W7sAP~GFnj}FRT8FFOeRWW0ae>7O`<E5"
        "2eBL?{;dvGnW`M@(Vu4mv+0A+el4pMcIF+8&6-aIZw9z)5}LpdpdjALbp{H2cT!^6n2DA3RbUt+-f49k<iB^))LfAx&lV?O+%ZFN"
        "2JAOF?t~uIC7_v#r#R0Ixvq`)>S_pi@ZiA%@uL-6+t=RmOTO%tnH`x^EtDcMS!r)yJbV7r3*5GyE4}c7t{AO(x+bgHduNve&o%@c"
        "h|d1_Hy!gc3I9mCw1vI(vg5K(r@%9MzX9&<^$@Bzyfy0*jH)Zy+MEe$=DE81&KP1n&n`>J_gUD$fp>GTfPh(n)$<*;<LLW|4Hx9&"
        "uToZ`H?~AyY7W{a1D^&<<v0fHsDKD(XX2ddI7!@WByK!Q<Vcz}C}{oi)Tp0EtRx0Lmip`ijZ&tUJ%U0~AgzJ>@ED=So5b`<+yaeU"
        "luBe*@U61zYzdMhzRz0RQRw7b-}Ry5ohUH&5ISUuF)F2R`sAz=9kHkKQ6vehL)kZIHlW$EhE+Uw$sM7Mut2cvP(hi?3N3~p!pfW2"
        "XN(u1+uo#$tC#O*V42{ysDBydrP%>V_CvQ#ix(i?*h(zrB1g&p>D7C4U*eFcv@WdR)e^uwpv%mn^3PzKJbM+3pV4UV6^kjcsU@1p"
        "9@nWHC5j_q@J&VRfe&e?f=8g6t8_=9VAd_X$cVG$9|IxNCCnU*%=j1Qf&^b$XGRH_@FCd_rqtGC<7k`5t|9DS>Y#u=q_Ga$tO749"
        "I($sMEc%hE54p1Hq=y12>W)TMwSKg^pD&kQ_5QJ2b)}&zt4<f<W9pn86IWgXjq4(0H5wM|2JeQ`8QQqXuWKuoV@JC~)bpan$I!30"
        "YR%bi4`pb{st{rD!ah4Jc<fbulufDPh+{YDjZ?~CI*1%%xE;Z|A$(jMb^Oaw{tF!6zCqJ!p@VlvV?(7HX3hM&{Ki7@T~UEnNmQoC"
        "L}$Kn<+}C=%Gy2CJu}R&khi<hB>JAgnT*duLmDujBh(fyeuOx9x1@_-@YYD3j;$~I!{3y8?=Gv(+vOW2P0B9+<byT+65ru3n-=<B"
        "y+Ja=VT$OkUa*g@Q0!BkV*Jf`dn35AL)RrDjRLn`@@-^6w9lP>!*q*4&f9T98OX&`lf%}5x@x-xvA&itzVt!~*fD+nQ#?%`KYpzL"
        "?{>N0y_z<kVX`a2_cJYs7Wce}K!1q22iH;Z6|yS#g23}lmN)7qOU_n4?utqchWqTfp3+)$UxV=qj3}`E1Y|~YrC+NmvMH^r114It"
        "Rb`SbXU2@3Sw^?MeS@x~1h1PS_>_G-H#L@y)2~ugsYH-2sr>vi`+{73$c;D4p#&_gTP@fl6zHbK;&o}FVJCQpX>UsFgaYsU(F$uO"
        "o6g)Jy;<R}M29+byjE>?`nF~(yGc`F|7QQ(+_cTD#kK<n|I$Qk`dR(?@yTKXrtJbRB<X`V=Bf@Xq_gR&j|Xr28dfgCL&=-CP3NXk"
        "oJ>V~A@-xMv#-p3*8mOWjq<nkrI0%9$|Iqrq73T{(*ocH-o!Q6H-<$$f6aSFV>Y3GJFaTb%H-Qv_@3PVH!VHWa%FqUiiN@fPUq@y"
        "jC{=fSR|b|2Q16|0aL3alrPOh_H_6)7`hnD);qj1qGrwSw*US`edjo(bazeKYR8iq4TC^9(_06fJ13u3+!8hH+o3dVc|Q-&{=EOy"
        "U&G1@Tv6c*j|zKNS_@<km1Wbf8TzC53=OnRuR^?iCOL3(iZiAKexwq$-`7my^TqkF7X7&0jwXipZ&UwG+9YW~M`sQ!|7IMv;5B~!"
        "D(t^|$vTWNiCV*Jm;U$PlH{BPt4g4ms@ow6w)YUeZl!PM+j6~yEA--;<oAof9hDTA-nXChOQrS|+lJ<vflt_5^vB~1&x1oGUC)hk"
        "&;AuM<U{@+6d_H0g-S5*Bv4&qP3P89giiMA%+zSG<uY^6h3HQ7{^y;?z55d3dfC-Bt<w3!Sk|m(qDS-{qrHjBE6`S|Sw<Es|5Eku"
        "U`J@>X;&$e6=QCuuw2IufsUNOiP1bvSw%W7=<qjP&_U#JhUyQiXH}-n)876WjnxgUN1lXpc{5stBX+LWZNCBHHBQjlO)kM~0&wag"
        "Rm_&s%#&6bpiVQ=ZYPon)dakXZ#HGRg*R@|0oiM`WHIG|Orq0hdj6Q$o_njZ-(f4Vt)3ofuveuv*7LGh>m*2b<uJXQvW3kO_VmUX"
        "5DIF0wfv`nV)C|LI*WtudY4D%Q}>_7in9&|Hr~orM+X~#y5ddEl$viq50v^gzXc&?X5a}mqO%}Ar_`jqvoYC@{$|E$5)8D@tQwuE"
        "<AUiNq2cVu7m?a7cEY|RSlw+H^s7cBLk|s{>sV>r{3ttZgKh};cc6ZkiOQ%_I!Pm<d_JZJuUE1H#phgRrARZ^<oDiq>tOHa2#De("
        "^IrG$@0DzBWz&Feokwrfxy+)hYtF`7^?zB4<%0{xEEr3P_iW-A;KCNo+_8X9A3fs4F+ReAUl&S}ZH3F01pWOAI~Yy6gi)M()VJEv"
        "AI<59r{J6`BX%&|JGJ<OadrGyU%#Z=U>qnH{kaOU8~-JiF`l<96rOKf`hq}Fwfjhcc6*fU(>3WzAs8tL{1Fs|5tdL6m@Tr(U3xs-"
        "bSxT;Ht`-`7le+d+ZWV?D-rsfExn=Rk(u#WsI-H24fO}g;%D``+rl!up*IsN<z}9MMY%{4OX@(z8~URb*cOuzlcu}mUk$~;=aFhg"
        "7UXXoaE)_A5M4=&>4*v#!C(NBh+al^OM0Cu3@6da30BYmXp-JH>My*Ej56rU_Rrzo`lNLPOHUc8pM%l*J3@~gZ~y|iX@rh@x}cUg"
        "QCb3er=(TpQPZL(Z2`m~%(dP)VZC9}y(IrVk|0s&rPK00vl;84uh73oC(DAqPq$-vV+2#atZcC<QDx81y+9+?pRUp_Z*_=wFEnM#"
        "pO9(4Zvml`C5`kS1Y~YfMAHg>#%jgLt6~{*B32C`24LLVqNyWF+l*lhf)jpZ&+Hmwe~IBX^7R)E?5i)ogpco<QcayWk$>6eaG#-@"
        "*#;EQ@_Y6PbMc*|VGF%Jama=kG@?H*8Gi3OTDracaO|FY&l#hx<5od{klxR*!D~2Z<T&KeOx1H0WeB{p9O1xQb)4e9aSv8_gyHoD"
        "&N&M;vY*Tb6UPtpip3paz77PWfwz^K;lB}N%hjF!1Bt+~38iKy32|$Z;QpN?nq7G}|NOK622e`_1QY-O00;m803iU;6&@}L6#xLx"
        "O8@{Q0001Ua$_%IX=ZO@VQgn*ZEayMWMOn+FLGsVWn^h#bZKvHE^v9(JpGT{Hg>-o@c+P5&{pG{ovaP#P*1vtUUSZ&XfH4>L9W1b"
        "A#f#XMz%&$L}@?b`hV~HASIIG%&ybqhmHa7NP2vH@9}+;nj}end?k8yR7E9+s%=DD3MtCI{aH0aj`vkv$^JpW<DqKyB5(U{)L)CH"
        "700$v^_MyPFVwxtM;yILlH~cb=g+ER*Y<;udYHcFZTB!;9Hc(f)y;JMUboG3ag@Vhy10{l0~71#&w!>QV5w`B<5XH0h;Q1l8C1U("
        "g(~G(4@H#^2D%)Zd}!Nxg01^8%VmC02B3qffDZfk9nEZ>-Be|p%X+VlN4Y8FAUFBgvy3K|3YF<<-^gL?)f&Wrx3dmrTnpXR)sW?F"
        "(*V7y$a)2`EA&a^ZCTb;1Nh3p;x|G!L)W5iWswmPfX~b_$eUVaC4i~E>#JrsIYIX@9NH%PX_P?NgPlW<0MJ)It4tQhYCz0J<TJ(i"
        "FR!K6;_nsU7>7T%xgL64mK2tF^Bu_Hhh8?iZ2P0?SKH^$1bkpU#9lS3uk!4+($$b^RhMhfokCd{{NS~?0!lj7r>l)Wgn(z*suXa$"
        "0k)f5(ZjVM8vfcAV+|Mh{aajiSf@X$l5hM;KQ#H`?l;I1T&-MU%p{0;bJG?N!6W>XRj&sic!JqoUN511N@)yF`}V%;S3)+$;`!!Q"
        ")}!I!o0c98A#G>S^Kgp<7U<MiD|8~>W8~%5;iOlf)(t&n9r)MDeA1)5*Pt|4uP@*H0hR}s8i?sY)iaoQ%+Vmy+*#<McSU&zdNi5#"
        "CYM8guxh7M*1M}8`cZ+Lw8-VSKMVmc(_jsXT&uFG!3U&?tncK5PS)c4hh~uXe`lzbKh`K5in)`$W<u9uc<2-u?D43YLG$$fIC$1o"
        "U~Q`J+a3^n*EY(B@VN}e8Ffwm1<g^b;Q|(KsPYRrf}Q^nXim_N!r!88hOBB1U<N~s4t1ZaZV)N+_!g;Iiyv{>&G)~7JMJ~;I$e_x"
        "iRRO^O+3NsrS#pIvmEQ19)-S!EUy@konftN@_GcHC<pKkH{+o6)wi<NYULn9$I*4F0EA0#vf4zg-stHKxV+SkCxvxjKxfn=MB1TG"
        "Bh;w>M_*<M3M1K)OxD38Bmo}bpYRa`ybKA3$M|PC(pd#U;arEnR`mf6ymwk@J>HOePVdl^)V}Q>LIV)sn#CRPZLDeKApOK&KUb@^"
        "3sJE>;6L;S>K5!M`=B1uzP+>1;I`mum*{rvbA?ZnYjJfYl1AMRXokT30&kjt{~itsj0LR28J}F6OSIZ*m@YA_PZ#JU(IWxj$+cPE"
        "3Gp6Aox`!3`RRI9s$QYHbr|{r@(&#MPUasFt%T_W&??3brd9H2vVwoom@q+z0_)C4bTO1ld@Ur{)?EczLa!}GNXGz;VE16gYlxd5"
        "(O#6Or}YD5N&N?~A~0>ZTa^`j|4Riq$wB2Gz^L@b=hPAELjJ$I5{CN@xF{imp;RGpbmXE|O-dtgREcVSy%HB<x+zQWzHo0HOk)F7"
        "Ruv#ve&~o4l7+p>Fzu0hm9iApV&Au8XXOhC<$oyEiK5cE><i$M<sqs9WC;0@(n8+Ksz$RHL(6#=<*FEB$+vP=ir66a0ne+!PwY&f"
        "L%RS{(K2XWNBa!$1~gMO1u_b`$&e<+2veY;Wk!l7Q36HUU9bFPg-~mVID=W_zy;emB={wmW9gqiE^pxn7H%snMR2i-gCTeD6uFDw"
        "wjXbq)@R_1)tv*eYM%jdX9115oduCi+km8NkZ{-PQ?L=<Kl!%|1h2sE^a5UE00HGvMQRpijpDni<?*JF;(jac1Dy^qhZH*qXNo@J"
        "vxVaE4SWK3CS6;I09EYMWJqzYShQ%|Cn<0p-hvcIdjBInR)abQz<3LxtvO*b3bzc&4Czp5&E1xnlfX@?2U32q2LIabd!_Xih7oK2"
        "wJIQvJNBAmwTA|5%D&V{Vahu3*+(1FFa+TZr!-9KN)JR7LNHnHA#X+d2G_47l%i|n$yMPIGta+cxNGtIEAeuE$kfJ-Ibz73Ds4}D"
        "OmB>_!EF!+BaloZLNZfP5N6X5V6Fm4M;#`)krp(Cx~E*KJ)*w_g-{KIHmpzW%DSSf%4v<c{K4_!AH=K6%S+4wE(70afzyu($YhT^"
        "Cr+n=6J~itis6x=CK=j6VuD7o$q@0$P3u6>uc#ZbDIl=mm(;Sb@ZGG@eOs$!MPYL2K|)DDDBRP^NtMz>VG%}D8dZm~sevGnDb(4F"
        "u)utYo8wYkh2jrGDNwj9r~L4{J_TW8vBwKzg-C?@9Fp(oKaYmu0~n|Zn+PoGAc(xpsFpMAf=u@M@u~a)pE1*v20vjEO*ySCF`qOG"
        "3z`&_zwS7{r*w5<7H^!J-b?nfw$HFdB8RMP>W9b4n4Q{g=^W6rr=He0_}KI9{2ct+dM3QEmTYp*ASL3EgcePRxTyg3nZs)*D;gF!"
        "wn(1gvJo<|yS%n?jP_6-8M_>#AUzH`b7}9|&1cg0a-uI78LS=!(J6#3sw=||dL_<p#UI5>(k{G3vLp}lavMP#&rj{g*Z1R1wIAD2"
        "XHeQ?y#BYcuOzJE;|j;hFsBSOCFekM`LpumelCZ1S_tRactWPU;B?M+kW(-%tssyQ#}FE9Rus$DW=^NXn=dD$_(HtJKB>TN1lAQ*"
        "(?LnaZL6XxNf6xs6@vu+;h6+y0b_cJGKDe)qToUG7Ts`=P|$&&YX+#@@<ZEdDp9cO33;6=@NU~fuoxh#0$T>nt!@>~2-Tbzl%7gD"
        "LrgTQ31n@!bb~Mr-bLCZ>9SgmmD7_Uh3<qk>cFBo;Z1n<m^T3?4ObtZXpVu!E$z&pBuo3%tjcIaMw~ouP*xzeh_rTL^3$Kj3S@CQ"
        "NeyVV*c@`hwBdZnn8#sNIK_`iq;uShy3rs3@uHv`d}g+GFi%8Jw5n-W<Yd&DF~pSR|EHU>O9RCm?(96lNm((S@KO%77V=ln3}=<+"
        "wWm7rtU3e;8q9O*>Sx@kg5Z|njvp7JOV1M|5GM#&N9VWMt}}l~N+9>oA?3VzoFw#=VJs-4J;JF@-}$pAR9>5Eew&Ej3o->-cK+se"
        "3gYEI^$TXS*S@OP@<!F`shm$!%Z=2io_qm@o|-h8Z*#60*)$_ORM}C>VCs2?8FG|=tK$i+GR#CA+vY<|#h^ZfU{5NHTQ?-aU;*dG"
        "F)uxCjb>v5Hqh3$pqiL6BrzL5+oARckSff5FG{<pMVXvl_&se6n~n-^Bt2zfNA2wBP@^yhqQV(SnH1e`_dIzbe}BtCl6`<``w@8)"
        "B}hS5tG&!0GIFRi+pi*UM}(*4Ou7VSp&G3+?vv--JctoLHnx=x>14I98t@pufrh}~tWbFk8Do)QQ^5A@Gdq2pqtDwriOPW320lMD"
        "?Ol^ZfFFS)7_)tuX`uTkk!Wot`YHegLsPQP4;i&k2<ddDB*P4t+%H!DeAFSF3MTph^R&jg6L$zETflfYgzWsRe#-ky7P2!nc@7S%"
        "0+!Trm5r}AT@wKgbB+gImm{{^dc2_xe8}S2_rxX&_0=5C$o2@v;?l#^BC!WE%3g8N)6vE`?-P+GmFA2CC@wW{Sl5F|93VU2q<A<d"
        "?S!5P=Pa|X1wDCS0?o0KDZ2pg1B>3@BB8gh%-m)d%4R8R(<xmGN_M%Y>NJXJ*9PrGryL^is|`39E~revCtELT2Cd|sPI^Xp(KOuD"
        "2ePgq;XwodoK9s6^^zlP<p~*8KiWsR!RX{?rZpbA337R)Dw<*<jxdVSt`;msZ{4<+rY%G63{U-dVlEbL7cI&;UvOs#3L1lOBa0%P"
        "rwV|6!_KAQ((7KL9WGjc&du{3$ha_Uzq9QpBjdV06v+RjF&Lk^LE{O0MK~*$K+9emeiH6@Y-8E1-~NlWzQW{q+E<9?%rOLhK^!Gx"
        "ISN0Y+izH6i84fV*wy>($XVL}h4Zi@eiDtr&Lnz+MjqALZqxEoM?8!DfCd^+@-;gg9n&=YEVT{$Nfjs@`LM|aP^yiHA`CdQkQ<0^"
        "HltJC5!UOou<_a@y2}~i$(<!Z(1hn9jKNGEtVvr<lhBOvvf^{2mnGgje)%lIv01Q1&GQ(Gjum1Smz<KS%?wSJHE&c24mO~}ZCmrI"
        "^K%ZKpW_;N6D;g^aopT3@=|wav?3ULY8RXs9bUi=9WiT2y{1-?_(zsNQm9*VYvqd28)0k-14M(Hc+NCa*T_4jv|ujpTdj>XKrz|Z"
        "vLjo-Q*a(W+h1573X9>O?MP>|;{&XeIOhStIRgTC*IPjPUHvp-RMuuY(?nwrC~nZ)2FG!iPXG>_{bJaO?xzd)$=&6L{1nPQPEJjG"
        "JlqdXlH?udiAu0U#LdH?^w%()4gdpP)M?jZq)z!?Dm+5<pWj~K&Y8eI9%~;wn~(S82MKOI>?+!9iz)E-@~PoY1N^hrkMG^Tc!BM`"
        "Nc?H*f=d&YzaKOr@Qu1#z?($q6nZRwET{=iClEgh`6;;OwYZ`xiVuUir*2kZ!1IdJS)9MSibco3-I&Gt?4wV=<kZ9@I(?9@zWM_U"
        "GjAYl_(mEbccU6~y5K@K2X$Xmd!?aZIE5UgeG2&`m?xu*pI@?{2(i?iHM4|mGO|N@x4^C2uzsw|tWb3Y0T6u8qFOTx<XZO7eozg*"
        "EU-AhW1sYVKpZMOM0@6B4;bx_Sam+o3CI%kPDl-<lDHY0qK0hO<k>J7v>06go~p!f-QrH#{RO>$f?FbO@0{>kzWe~jRaSfZzQf!9"
        "diVBk|Nhsv-~Et%|L*NK-=7*4ec^+c%$Oo|cBB!9H=17Gm^w_8z-%|4xCen9WmA>7-A4zR*3)4oE=5(2zAWUzF)cnuR#HC~!Hv%K"
        "YZ|;M6kSESw>Suk8u+^**@7|AJ_i<-L|Y_Z;0HT*%(cUU{{R>|A%hb+d5^DZ;7d7apkA>=+mX^(meoB;Ai+~Y7g$lbMY<|+hjOMe"
        "*|+V!R$tNrUcA0|`|95)3m`m@hv9N-uh&}b509XD{Rj%olJ&Afl_nFb8+<D&Sub-vTogteX*?|vh{TZ|8N>7dOdM(@b?j=0x10A>"
        "^Ij(L{R)$0q3AflCnt|$FqgKHL9r8Wp^7cDFR)@mkg>B_A+l))5S)4`L3mSd!gwdo7Tx2`#v{#ljE=?@`T9v<7j!J#fh;gtpaz;T"
        "uB%gnU;A?y@Qj?}6}!k?V(xm7#!drB+DbEzJn$Q-6}fmHe!3kSGr48^WqCmRW~^#_b<LdH<hRaJP7>B;&wlCR7tz3T{Mhw^v2|hH"
        "f%`_A!5wT9&;3H)TBVFFa+(w{PsITkA74`?v0id(=HT$iu}%}(So&%yv0jEjSoyp+f$)%{i_t&-7|7++fxKm%Vq-g#r8Mvi2GXmD"
        "FJ;uO8P8%2>5fG7ARnBDE&@EPAQ$01SjUcijK*)Qd{LQ4U5k67*dq|p^KvheNB^Yl#}_XU!P+_V8j6hNLj_t{w|9wmB-G5Zdu$Lz"
        "Fet=MQn{Lyz;L`>+M9cxl}j-QyBWSvlu_e5S<eS^P31SKpCy}=9AA08HE)`Vt5a{BhPq30;OlR5ll`MT*J|;S(O_@=_S58bDTfJ9"
        "&U#8<K)(<_z7+U2sm6VwFRivQ6d#nb?BVMxhe{7^kKG@kZn4RXt_4DzMP^@k>D5gSg5p860`ogI(`=+noG992#AJvr8|X;!91Xb~"
        "w>L;DOgwov{jtArQ)?bRA7Q@sd%B59(6^?mONue?tT-bvS$9ix?t0qy82H7oqwFAFH@tH(H$=Sl!E^w89_s0v4{OwXJ6Vj^B^bGX"
        "On==6@l7e<5qGRQFfDv**n9EBHc9Nx?JO3vC7A{L%<3L{8Mq_XL)HKxfzH>?y$M(TwhLwqCi-xfxTO+4w5KX)I(J!2Z;m$Uz?DN4"
        "?-o*w)t>jxAkSJZIsoMdl>AFq1UJ052Oc+kBJ+Gaj<przYp-Y2;SQr_Z$gLx0$-@#<f^Js9a17@CB77|z15}HuWupn6ns2AGY*|~"
        "sW%Qj0}T7&Kz`oz*qwuTjQRvta3;S5Ck$Kp3u$bZaZwO6QZ^BYp?(^1L#6iK2SKC|EbGEOGDRnl1410HJ?*`ivu0d;2XqtC>)3%T"
        "C(KL~S@6Yv4xPu@nkjd5C=*wVuknSumn-oCe4iwHS_RvI&WRoiVovpRUtK~_x!in3XL-6eU(wYYd-;Yh-_Yez-oxcf_dr!(KvvFy"
        "8k?}c_1DKq2V6hLF<}wj0yS6Na302SNjq<XN!A&kTymZ6c3z272Jrc68YPcMWJgF!Mx=oD=@Rp1w+ahkJ3-c#88$;R@SEI%G5uDX"
        "je4%qMlzOYX7(d|e!Ufmdqo2>1=1Lz>&e(QUE8I}1Pls88GaOHN#|%d3I{J@=mpBNz|mPAMepdSIWqA0Z(gLd{}$w^<{SV$GKR%z"
        ")7m;dFds~uC@-@H6h<GW5Jb9fS`7hAdtJHPS9tV|0!;MOht&Wd>BNR<3P#0{%pv}A?Y%{QId7!+1%f%~8k8tO_Iu9MaAFUfS9u;{"
        "w2XX&Te1XmL=1_<#bEZQQy>+rGb+93Y6z&R>nXc&bN+=h^X3dydFr7)Eq#l2SAEZAw>i!_IO71whMlU?w_)3b!flNCCulg(Lm5m("
        "g00n=BO_^Qt=T7TVw#!y3j0KLvfA9Czsm5ql~W%k$>x1qH7Nqo4=k|RtYFIIznjFX>lhzc2=zKZ@2-+@C@=mwnOY7kZffmVew-J@"
        "fs<)B-2S*wzo%dCu>DiI!1E{e!RAl>o?gCsM^Dm1>AbHx<nJoWiZ;)(l{*wSS2JnGq{+pFechRU8RQCuh4Q1iyKsM(<HT?l>@OZZ"
        "sqh6C^bQ_d$@P0oOw4%tgBWx=iBpd?YDK(_<21B-{2>+Y>I8R<c0{^H=A~##PCz~BXoaH+yeAK-NImKUv+H098$>i$*kbT>BRsp|"
        "Oy#FRu`@DmHMcc#32MCf>#=SWwZrC~hvy5Tr)#L3hz&J`#@SbG%^&(ftx{wewpFu?j(lbrnnae_6Q67#&!7D-P)h>@6aWAK2mk;8"
        "Apm}@>yW_>004U}001KZ0047xV=rQ9W^ZF*Y-eR{ZDB8DVRT_Ha%FIDb7^#Ma(OOrdF5K`liRit|Lz(84=nqIN+h1{?NdgXWD+~$"
        "X(w@fj{D(y<v=8OVvize0F+L4?|<*^0wBT1Ju+#(w4AX|1Qv_MK7QB*===V=tSETm$+8wP_hh{08zwzgB;HS%@?^m(xh|ErVTy~4"
        "<=JPZva;~fQg~&W=2^i#!K>1{eDfjnegEZ)moKtSRSM;;nOx`Ds(JsXEQ{tv@aCtiSCuGZF6GM?sVFyIT;@5CaT*!2RqPqMw;%yi"
        "r7%<pQ!LJzlw4Yhrpb_0OrZp}`yITZeAKtf(lTcG4c}~77?*_-ELIJt_q<}6czwq<RnDjQTFMOO-oMGZ@0UhFV_Z=jV$JWAWgN4j"
        "EHWU8tZF0F)6|n3*Bpt4b7V$7$Qf1#)@no!UcT_)VO5pyBkGG(*~~K4y*HVPc(KhyS!{S=TQxOmOPUtRQcki^D>P0}ahJdR_Wp<0"
        "(fijQ-(J4`<>yOp>3z}kewy8>T5x~i1->ilf!r5r%~cl9+NI4b^|To{+aBT@SRpI?$+XKg;#-#gQP#wx;VgGNf_^TLt!|_C40JAv"
        "d~<O5x7V{bNHF_=GR+PT=YR314&?<uP7jaI&Q|;=zF>#!=y;wSogA-D=JQv_rza=S_3HF=e%KkL*JoNf!U(?N1z2Hw)RU;-tE`qJ"
        "kD-};JnAq#oS(cpPWU`Mou~XXVW%gj7bi#O^HX+y7M~r04CjX|Jx`w@iQw`RDf-u12yl>t_pdK!zoPo$Ie&FJUmeC5^W+d-R_Dp_"
        ">L@*iwX-uG&oAPmcz%+yj;im`YhJ$it_jQleEBmkmLEjT!KJ9piKY)s-Y&E#Ss@iGVjgA5!jnoEza*F%G=TQDnFPRadPJGKgy<s9"
        ">xxMZ5w_W6%4yA{tldG1i5sqTG&#+A7V|=O2)R^|j73&ycHY0dpG$zoCH*l`wpZ)2aKrCGQo%pfnczqxA0*0IaZ|G!Ee7=XdIDeT"
        "JmD#=8T1ESGPMsH4NN`2aacdFPO?~CQ4{=k-JNCP&A!EEV-T=uja|ql1lh_huLOq}3jDQFm0TPg0J^N}6+qtRz@HczS*Ju3LMJo="
        "zpM&;?9a;`7s146KJ|(cw}mjR^L&E}4}^ckW}oM?izu93e|_M-fR~Xx6A$(wKMZ~F7u~XGeH=Voa_=Lyz80bsfnOK0t`IGG5@}%x"
        "<a{H9Q41{6{sQh_?v1B$UO<oJ%F4W3!)9Rg0@GWiwgSMSnqD04bdAeL1|vREypMBk1+(*qEJ*^r@V3Jm#>{_3@7G=R{t*>FDRou="
        "VU|R#^j2=c0f3ERx^8LqI0XS8;r<qhau+z!J%xF8+CRw;$1x(cc?&KKl-%E-o^XcSqTChkmaDm0=@GTS+z>$^jRBY`;R<NGKh>2U"
        "du@RvLMTOl(E=`)23%V-!dPOv5CFy)z81NUYqEoR4ifvWgAzWu;}9i9O}Wy@P)Ms*Ip0Kpq>8}A@*zNT`N%S_k%@-7F;>)WhHKn("
        "$2)M^$V#}AW?XaxPDYqJSnqf7faQ4rdN=*SmT+o9>FL&B$%JyzEt;PoZmP93)r*`OVW*BflgXogv4QbfaT5*PG~;>FxHcx9kaao2"
        "*@;lXM?noDjLp*TXK${*oKH{Y_r4YKak*@mjp%wPjMhY>ET_k4XROO;tX;t4iyUxnH0<(ll1n{77aO{P?Ib=9x3<v<Nk>&SHk~F2"
        "(hfy`Bb+$BdQBEEoeol!bu*`3>&by;C4$DoO>*GV*3@r}3brv_7Jqz~b3^YW6hID+q+w@ik}DlJxJy4Ycm-RWlWP2aCT<QAsk`TM"
        "=xW0xDQ3+kMVa)JJ#jY&mP9tID|h5GYnuaM1xMR~tp%@egScKao`#@?L^P_6LVOE#CPKo8Sc^hM(F(nTY(SHh6)yrvv&#fPVCmN?"
        "oxK9-p(tL1zvsNS9U}Np6%6<@N?=4l8tY0U6!`2BV5m$6^{_++IL0i3=MOEzT?K{*Xux4XLrAAsOAgmH%nzJ^Ms`z>T&ky&0An(o"
        "Ix?;!BPImfyGe{*VpG3Awzi6iPc>KW@=T>$?SXH(vlM8W>J$2pYm*giabT3{Q}|93V(qRSBh0{mqLnV71Ns}u>@!~;9=+->)e)Hp"
        "L#V2&Jt`$=Y$Knab{ajUAWH;t!lvLG@7Kl9Pnoi33YXyEma*^qzkri%GUVu0u&I_JyU7Zcd%wN^E9`EYO|2O0_TDNlSMnP$rH8wg"
        "uA}mtQdz_WX-{d$Um)P3C@V4*MS<jbis?<M+;E$EUtzrf@fsCe?MiXG@K$A+gPXl!uy9vh$RR`qHVW}K76Oo7Y<V#0QFgde0_(h6"
        "5x>LaC5ty4fFp6FhV<P)gOL(SHC-zaK&<wvlZt~3b&hhir%j=#i=b=kD<7*As9<L^_%TZ|;nP;8y~a#(J?#zxlwJ1DRXV!`J~6B4"
        "Afv2RRVylgdlV|$aOedOoI5?;@YIV4_09=_+=rTh=M8bjZ;qW%<dlIMLVfLfUt8u<C0vLw;mAKQFfpYxG-sF_UoDR3^Xo3>iK}SZ"
        ")?03{DA(QW{6W_lrsS=QUOGU|w_$e_E@0iPKog$M=X2ext=u$6(1HKVrqZn&Y*e{;Vlo0#)wUS|$`VqV&=Q-_TRL8T;05N8Gn8v~"
        "bb1z84j}<8xcnqs^SdOw;RqL?Wm5+aP*-zrLnWjPIS1F@@<^3IGwIVvtXWaU6>$lk#~+@L-vN%v!H#5zhEY1Qb7=t(TYoOqo3btv"
        "BgbWPcC0mL@@jx46Y(hrIE5D2a-h(Yh+awP%4(g179MVHAs!(T@mxyk5b{X=sJyiYN*%$T74rBrVISv^%INslLc{NL?0DmgHS~R`"
        "hayPmA(K8n;PV`ez~^b4z{g?*1mg6;Ks0uI3Nu<0K8hRowAev<`j4SUPjDsnJsF#Nb3`!m584g1`f0N54BSK-OK!Qy(!H+{CUmMN"
        "499<b(ZBEeeus};$5@w1<sMW&zl(XL9Hsuz5E}`L?y@PIaLGC|7+LV-e?YAT5YQ3bh=jSEDZ#aTBLCdh)uF6%AYA!>pSuFlt^%yC"
        "9EVgTq)X5?@1~PD?;g`hBfz*diE6}vc$JeCb<c*7(=eT4V^s)q((qPR#FfLIDmI|Fsdt+d$<!<HU<%V#7;1T$(L)@-A>N;s-pa=W"
        "(Z_TUdD)Y*FDw6WK0iA-fjQ~t`RS>D{~WAnQ31-8TN%OD1Aq&>Yfx?k?4u_6?K=bN-z>d*;S^wN4@D-u!P-z?FLCz|Pz%E~Ym#r<"
        "Q~_5FQwC)CiTCZ&NiKfZQgNTAJbqOtcte2E&XeH^=?;&wlM#1~nVR9;=&xiHGHDYX{q>G|7Tql_?L2*z^%7MLNxu~t68%V|W6*@p"
        "pS>US_2a?in-5-%XW=vzl+AnC!6`2RhCHTYL)g-JB0YxjR^?pboiJz~s;l@GHmdNs5dk*C5JiW?*`6IqWWkVCP`W2U_p&yyPBtZ$"
        "lVPelOcC4suRT6gb!C{ziE~_00TpB<?qQ(-#fl`t6NRXk?T+TSy30~7M&vOhs=Va|IJ!L$`^x+U--&OzK7Yp&h|`shO+%h_*JEfg"
        "8qBqGM6)%?^~wC=46lJPrzTa<$;*_O8dhW1;@55Y`VkK=8MsDS#GH-}jsDe|dw3O_v|k$_;sS_B4rMz`uS-CW)k$XSSK}?(P<SR~"
        "am-$9niHz~XC0~bk9v`#a}wPx4y=^AKtN1=sCamEZU;+XG$h9x_lVu52X-!Jf3;_4uvapnB>Gt}*y*1)wI&`+-h8C+aADNkAx=6z"
        "h+hN@4_jMevMgPGfUP=ydp}}RJO|R|B!6keY6m~MPP#4MFmdY|_*2aULl9{!yy74gr|&QjO2@y@Xw2T+fl!#(4>u45mk)o$;a0mg"
        "sP#h4aeN!w6ofkXmcw{#&Pz2&|5WCQ$I_JJw<}v6VJHf!OJ80SVnc?(Az!mXw=x@K*5S8&=;M)~C$oDk-Z|wrD(n@-gE6)`ZgzIJ"
        "NY?yBxof{JAp|k7s`eFrW#T(J)^l&WUs||#N(-avw#lV5mwtvwzYY7FPYk+*(a7b4Pw=s59og+`4#Z8reecU|Z}k?gn>aMal+bTI"
        "(G}TxuC@RWIX><Z-iWfUkcV~p?i6&#v=~L9&U(=n4{@>gA$Gm`bz~_~<CQq2{p}LA4vrq6cLig8?YWq3rgcFBA%N+;(|X0@0OR6F"
        "xx4aRclEEq6$eh-8$clx^%Z~p2%yl2$s_Z+OxpMw9<`Obso8*ur6yQsNkShieEF{g1>H(p7e|P9`rHf3r=92dz6w}p46+!Sqw{T("
        "b9rV-+a{NC_)U%-Fzo3rzr7r5)zG$eea<zMwo7BGC-cbvhs@E*65l~(#Rb$Bm=#v_3gE(1yK<&Ka^ZJ$0rdU5x8JzC9Vi`QSeo-("
        "h_p6G?C3ngK>A$?yH&k=z1damn0g1QU3T6aDx>11`Rnuy6*`g)wAZKOZr1u2*PpVQZ*8<^0im|-^l=};LwGt?Xs=f~-^aFJ<2;!="
        "QMQaYHvr1#C9KG8kgm3MTq~1fKbp@)KKFoD3!}l#ZKw4<ZuN$*-Ku+h&0V~^kvR9={{~P?0|XQR000O8001EXL-SR-Z5#jq#ApBj"
        "DgXcgb8=%ZVrgb?V_|G(Wo>O?FJxhKVJ~uJaBp*IbZ>HbUt@1_aCLJoaCyZ&{d3zkvcKO<{|7$a`+-U%+D_9qIk(EaT;lXRmnIq8"
        ">9nKxI1~v<tSM6Ehi%oD|NZSQJ_vxIB)9kGbk{^87mLMWv0qpqzVCm`ih`wH!|ogJ`>gTeDp_T>toGs}_3|uXMa|N&m(_L4CS{Rt"
        "y^3whI%~>m>s`LR^5S$2FSD`;ec%7;Phb5hTW`v$@m6uY%CmX*yJ<FIl4q=F)U#jevQR%NroJ|DwaD`BcU$FvDr8kvR`$b+{o1k`"
        "a4xEH?ImTNv!nrXYEQtuF59AEl>kWNCQkCWu9*UW0p)SR1i&V4kZ$?%Bm5kDAKQw3gi+t)PaGtH&a*|C#QBn~*KwGXMN`E|qtIR`"
        "KD~vpjJ4mFMZBq3W#dAF^o2=V#W)6uD){P89(+EYUHou<b#?Y8di~+eS@iDw_1XK&v&%7m`?^fohg(+V@#efp*}Zt2z~iXSmPOpa"
        "AjCs<k9ea^S?1yi@;geiC9qOFsW*AnL@=xZ7|PP9Vo6!0HM|`S5?(GAc~(Gwi$>C(mvI`=fB*+|=+#X;hu#)hv1HYz%8Fr5z-%O|"
        "ZmQ^Kgi}-lns~t?lty|mjAGlY$|Cx;jbS`ni=~%8zrT9*Q}pKi)%Wi|Twa~OzKlM80LK2TxCg=$)!U+p@1wi8DuCu%GI70Ww${6-"
        "ZlU99D?qZM-T?O^K?BJ3k~KRysI4xO(ha<cD?VTjAZ8@G%bHbG3z-rfR}8pbmu;0mi}J2kkAYiR3TqHpp*<4_o4CG-kT7h+$;AKB"
        "dhuRy%^-#M5?ZiRjsjNx1$2lC<nQgAL=LCji$R~juk|`>zV!+Q!U>W`tHdkuD@)pj=?Vo~N<D&zQ?~G;C@Vk^Bf$p}O_ndl-Os92"
        "<HpYu?R%W(<y};;=B})6rf=iCX4=~|Oq@`7(R|ymdU|qv93AujEI7$v0r<@j94;h$tzL{yO~OMY*PD`N3Gn*}CcQ56TNaEgNP~K4"
        "V*IvXdt(9k-ZMuVb6ld4`OwjdMk#>lyFO#GbwKmvp;YNikn#u34S5U@kQRW(2d<I;RM?ZSUd7M8`JW(ASe|<R|N8JJwVuZ{i@3y%"
        "LRO$W1*1{8V)q=8&SKOWfVi8JBQFr$0nQK>Yq<Q)*BWFCuP8|gtwMI6)lD7f<I+I-vuDAP*v47SJTwmP%jPX8n)D0}Tp(?-WTUjn"
        "V1zxgwZ$^j$ua~hT;HTw6`-mpntFOwwP1RP*r>drpH|C;p_&1e1+1Wvadpxz7TJB^g95IaEY9uz7O;Nf2HS$7FJLI3%7d?MZV>)z"
        ">v)PE?8vg>A^a>itO(pKsAnHNiTd>DNUE7|35eP`Sm5>1gA|J|M|}Q{9&EYz;*W>1H`$t%Z8LrT%>dBIi)#?udIKH?8yFZNDw{Y5"
        "mL3m-(D)VmLmIb#H}|QF75HEIt_}#p3WUc?R&8<PolX^5A!`5N<oMZh?}>+hx#q=9@YAW60~v&Jw7(IEJ8$=LKnWetF@E?`_%Y5~"
        "Cf0z&l9qP`xe)U<1=|_1`-HKSrM5u2KAeZ)U6nO#=aA;Wu$%rX>}#V8a1VRu(1w4%y83uV(BQQVNRIBrhMz9p(Le`aueccQ0(s3H"
        "Gw5#!isXp88<+){J-LRbQ}5Z1x$4i4vJh&5z^;S0gKp&_&T{ZIL6FJG5pKJVuakP=d<Up{OjS$D2wX+b+3***u)ncG#q5uXlX9iN"
        "9RZgmUNW*&I7IMXWXG0NAhp&USbmloyQPn;*ElxrVhosQ2R!#VOyjJ9X=9Tk3fm31n=Fug;Z}93MFD(9#-8aPIVMHJUNML<3I_wZ"
        "ABny}pO3^j@&xctG6N7P2;P7*r*TAcx-4HyYB=ea%&z%wOAXhI6=d?2BT(ODn++J-sf-il3*n``FAF9^0i?)y#5Fh%R=NJ>gYm0>"
        "q33|>PmEF%P&A!uww{AC43@M1o=a2J;4qB1GvKYlB3`o*8Rps&IO(5YW7P}6<D9`$BslPofHZjRhoO%o5d5Hj;#x&G%81oTykWK;"
        "1{#QS;Hw~?TxdCCsn^-@bss=6n*5oH;?TgK_+#&6G@G1Uhu~Iq^W<Pa#ll6K=W7fW1Alfr`Aa-mTt7TN2GcLF69W*I|J2#z)`GTV"
        "4ihnCY=_y&F)BaU!BWXA@4q-#9yMF*HgB>`j)7p=)DG-IPO=(3M;G~++sKNRImJ!2##P{KF4ox0&$HsDj>;-3%A#V+HjgX+8hwtF"
        "NA#J~G6Q~$yziVwYX>K|_Y8q^9yd~uZH6dENDK^hT#pFp^(fth@`RL;0vdTQwb1Aw5|xg}SdhZ@Jr0i?1FRER(BUs-CfG7NIlZ=r"
        "<n+PK{7E1YT5W|B)X}<Yg4?2QH|PUkt_OMh=-HK2gb7&4MVY6;2z3-H?#J;l;T9Hrt`w)7az_2Ai4|d-rh%hZ8?*t-wZ<5n%Tg#3"
        "86@;vQmhu3W2_q45cDA~md<wQoPc$YR*4|F^RCGr1fwVYHA`MPZ#qJ)pfz~hlAv~E$Z>g)&g`uSr}$9Hcq-`+Q%1qI-!_ZM3mbzF"
        "qEAJJKyL_iuP}j<hydG*#|x4F`IsyllDMmfc~*cA`bhk=ro$L86lq{4d6gWEscR`iLQx94CJc|Q8v0Yj^NZvwNvW266^>Rvr5tcS"
        "N_tOC3wmGZB=F@Py6*f<nP<t?KlOZ+;~QA8pRV4%h=f|^-D18!LT@iiqRY2edf@C60AU6ko#rz@1Y%N<07Xhu2M`khuhJ}O0`b$p"
        "v1khla^gHnRxG)Js0i&P469+#PS5Cl(Y?&1Tox$PVhban`4H@`H|agO^kL0#9R!ts`SHz9lXo0;a!!VLk+JIZbNwWk{qu8i{dDAu"
        "NmX--XtW8u#i_$(43EQQRkoYp1fAq6+kms&a~H%E@>YONd?|*q{2qgmdZ1x%3W#BRj)<^ApGcV51z-&H*=Cj4ra?`#VilWjy(L6!"
        "4estND#pBW;Uz0fP}9;OWLh4zNMSx#%&W^qa~D^XjM`M?ZHA`FYM}c7VwgUNqpv9Q<4p{+DfgGU_Pig68Hmzn;c@rMfs<pIAeeK|"
        "6-zVZ|GcHm6vW-u#A9w^=?T{X=#{VQ9V3U~{?4IW4y8Y@)5+J9B#(g>zo)tazhK~n0xOW)=(=m%L8l$GPFcYy;5!hpE*lNN*19Ay"
        "*x$2$e|zPmDUn;KOEy~wx5_kIz~8`*Bj{Q&mU#wm20573$suBbV4au{YWP3NCrw|$iljx0h51oAAE0#^*zFYX1JTI0O*;qK^3rl0"
        "XT?q#Y6~fZhGx~)S*KGdNuMf}f|E0r_HWD;x*U`5`)^Ck{{q>#+Xihi%#=>F&5n!9E4c#TO_H}M8+&|D2Rs%p!>vRS67W3{VvE=a"
        "!=P{@HY4$QKeEL*GwPDFFN_rYf<$Y;v+A27L1rY#jFN4V8=;@_q$oJm*(?-sN($H3%F=@|DAU@2W20umCX!DSpM<W-*xo3R{>h%r"
        "0L+CITWhVZ<LZWo=@N@G>Z6EWfjR<6A6Dw7%2<Qmh9gtwFN@Y&vVv6*R0!FXVmSm3!Uq23fvq%x?EuZneTo^bxqVz@MH=Th7|4Hq"
        "4nK!p@YhoT@Gl6;v19RbbPA^OY4G{-=?F~bfBV|H05m_Cj7D!k_CwC=$!?kwSo#^Jt1xXK1wSl6V^iP)nkX<6B`QOLj`fim69mw@"
        "TW`fbsuz>ku5<L-V18XOokg&r8PG?TIE0p%sR!4F*5U1x#6MP+rKY0vj9jt|?NpJlpv%fpETn0w*XUR&RW!$)s2HTRCcze_QtY{d"
        "JTP>8fr@;KF2fRnB~R?qcUnrO;T|Y6hHX#jStxmenDGH=AVFA2t-f#!l58!Phi&){I?q}GLE1EkH&c6B`EzYffwKU`y?A&d0D$e*"
        "R^OmcNcIG<STdmIl=6=XEdY+RBp2G`oeiyom=Iz+6(~j}WZZ1RcZG1P&k_4za7!+u?~@PP$i2c5?g99r&%Fly`BO|V^fBe}mJrR{"
        "=`Pbe4s5*Ol$wi0F0c^P%er>u*TnSLSxr8$0eEsjj(oX<vF(gv7T(A^b)+F>I|uM^@{fos^gzr0_&XH44HGu^%|USM^dPXRJ_y|e"
        "o%KWDxC4GD%0ovZ(C!6C#}pi;Ltu8{`qF>wkOnkcDDVTxU1JT`lmH6|{&72w&J{V(<p2t$F!-0AhLFh!MettdPi9R;UXlvRq!16e"
        "db7j4Ww!$8XuCG(Mh8t=DY24=f@N}k$I_;ag(Ivew4ESF9<e{z2>Oh$+_7oQbgZ;hGAU%@cIX11a-*l7st4-<h%PZ}GN4}{Bog5?"
        "$Qe292JLgVfzm7JBRC`-KiwRtmZA*C;F)8YSd?q{%{-N4kl~^v`+SsSC8tTC(KzwobKXUQ31}^5oFms`gMIvsdT^}~i37*i(w*2#"
        "56>b8oHjAU>uyja!uQQCG&K|Dsha!sBdZ{^_OLu|%blV)X<Bf*w;nXQqevWsTJH5#N4R&4RhHkvBJj#8hER-qLt2!$=z1b2$45Uc"
        "f_NSTSLUKAceXPFX%7fD!7^CcqL4AdxI9uF$t6LwbZYS74xGmFPR<T-oj>B?r$HuvQG@F{yB>WJ2;<v0%duqC@F@5DkXpS~Ez_<T"
        "AanoM`)=A{eiR#&LL}OmEVQ(evxj-%dWB_|&izQmo>T+;j-^Xwc3^_vv=y;p!>UQyHc1I#d{S1Mw)QANx+$}wslWBolJRM-u}~B9"
        "H!R1>+h(}Hl9%i8Eo|UZgSD<k)_~_8>R&PT`&y!qQ!Yyytx67x)r2j2Ol=xfGo*8rBi>&qN|Egj1Yju2Ndy1&#o4Q?Gw<rv-`<^h"
        "=Wo6DAFjN!pUy9@F4d?4RS@l6o&9t*@^|2~(sw(si|MF?a<PD(yz}>0XWyS)cpop$e|UBAv-gj)pU0bcOQo^A+n&NC0u*giN2zq6"
        "W=3m)Q=H0LMx7UzXYbBlUuh`cUVQjLvB4jO3kJHZESQxG5g5_`<zL0-vqH4jfOf(xN|$*>yX+cs%`m1kbQD<f9<3gFLpujUe{qFO"
        "{?K9l@zuLeVE%)@V!FPY<?+Zd!F-d12hMUWV68N*16}S>(eNq7)xqbFtFB#wB3LgP%67zmDJb5Di#H%2fBRWL>GEI=f15D~v?UdI"
        "a3Vx}Xp9K4F^W3AWnwj3boYvphd8Q8R2&_1o*E>i?RwLZa43_LysTM|%c=#v)zTUBHp^4J?wzYU!Edg6Wi2~QA8ks>Qw@~MK*5VZ"
        "^N5E{dBf17siFkq40+t4X;Va9EV89|1U%TZ(mm5*WrevggLFs$OUG5sHz;SGg2*!%{q!68yt5?u_{GUzo{c42$hM0Y&yJtJkT0_O"
        "MwS9Bv-$M7L^WL5zNyM380mV7o6T6oujQn{VE+dGg(s^FCY0@gmCkwkt?)wFeUi6ww<s1#X@C+^(N9O`af3T3g6mlI$KPP`YR#G$"
        "&o)S4vArWD!g-ViCP9Lq{RB?_`44CBucFJ3XRj}ZTG8<dF}KF)8LIcYMJFo*Tdih)2u}mL43BbXM=|vCEV6z7Ach<es!yIY6G6&@"
        "U*Dk)zh{+?Uilq1o`XBEif*6zU-tMO6_b&dEB?Upova9SNgP|^H#Ws?Yzo`h7BMWVIoU~5R@}yUmWp*Kno5Eo>grY<n55qMpTJy{"
        "%4u5SmJ(K>ckgFDj-qP(>zPmd!Yn(U<iU+g@PK_0AI|d<H=4!%heQQ!`15Y6GK>IZ<zL<|>t2(jD(f2A!(fd1922-|4C9Y$9_(v+"
        "gVP&{my$ZJMx=Yhe|59Rf4j~Tbms%k=%q5Oix%Ba3=6QllFFOSLo_%o%CM5+lD*Rp{l&pP*`GZ;;=_*sVyPoi0D(H?5^@&v{mMAU"
        ">Xq$8?$D>YEMV^VJ`@AT?^JYY9$M26`M_Y}C9lz%67iPcbP0Z69+-;=FRPD~wA3MYANA}H)u7@)_;C3PDHY@`Po1ZoIQOtWc}fzN"
        "v#E!TUveWZt7@$4TA)`Af`an6fG%vs1}>J!>{sLd%M<ldNFXmSQpoiRBks3#ef5fuR5#*FgxIL!bU;}K+8}rW+BBll^q07G9Vzt2"
        "&;6ZrB=7D;D6meG?Zw<pzKiP$pR-d#K1jpX{c>UJ{2u`9Ivyxn#R=9$@r;?!7RvPMy8)zDhPrkXHA{DRYLxtlF!j{*nY35rIz=jL"
        "q*sG0&&sUM=~ko`qF_@^hwE?jl##C+EtL%uHcSIEtr-AX=1l`N3>`q4hED@DOdux643nq>n#PfGX~q*BSe=EAgpY&eRhkWvMR&K)"
        "rCg+0ye!JP$&%WruIj&zRKeE1GpihdW|pkEamADxv{}NX)&gH*`%#%;%gkK+ZjE5CV=e1z5L1C*WNQ>)E2rx7^R*}%!V#6He#JZl"
        "BP&(u(`gli1OR9!;mqg#R+1fNo9?OJ;!9VThJWRh$blufBh906PuUvYuJ-YG;;wW69a<E|Zqy}k4DFT2XxDjmV(Lv!WR%p!w=$)K"
        "B{S4<NFlNAmzYTS7maSxQZt|3$KaTPqiR=II*(Kb_W5ZQYEW8Gs0OPM&k~yX1EU#cNx_Bi?O<9}s|e_sg1k1d`P{STJH8yYSB8QB"
        "$Onl^G_m%Oq^6g91RSbY)2-C}j01o}8W&}eCDIuMSNQnaRj?}gE<18owYqt4!r-Sj8=RW`lQC8UCEv=#tZ$k0x4}EZCwZh;s$Tha"
        "WxE`8#eShw>?xiEcC|y0xW;dxI8buxk1ST=K?@N(^Xo4(L4P^*jx~ZMT=3u<aUd{^i6*!c(;<MkPt6Rkmz6(Nn)}b*B`t3{A?9Jg"
        "U7_ceG6<p>#!qzl29Lxk_n+i(w#Fh1&{-*7T~Th@T%_4zaKJH$?XU1Ero=ldfK<d9*mnhJw%)246TJ9ts*lXBX6O}7i;5pLUgV{2"
        ">-QA@?9(SwV}Y-u?d|jps8w8%se%q(dLIB^xO`^?sM!LTk)^&&py^tsUk*<bHXXc4ArSr+vfh?oxw?lB!;fXK**^q8FnL@oTRfzw"
        "vJJu-j?pd}=>_@ehxAx$FPinc>4*gMGS&u(QqR>XPX7pdo%sCB21d*W_1?wy!~-qBcbqQMbfhBS8fD6K3lHj8HW9P-Yg;yPug~tf"
        "0*Cnx6b<A4$xi;H+o|k!4YdI&{Es>2lD1e<3adAwn|UI+pDOI#fVq=hS?~K+5XLHDEmW?vu@p0h>edIk@4;+lZ-eo>26pcN_jzYq"
        "RNN$$7_<$?KK~w!O<=JH3B#+1={@4~ePMzc*gwXPeN;wi0$LCFQ*EnHeec!#H=a1FWsz?S2428IxwPz_!ovfve{l?+Xy`w-E{xGX"
        "GJNm9F%oK1@yh`xG}cu7?SR13aWXn1|GAzH1#2!gJ5MCtW_5<^(;;nxTl0M^26J#8W-6@gwz?e^Li^6m?K+J=yyuMm3|c2@IT}7u"
        "KW7Wf6nBM}vg*<-xg6XpmmiY1gU}G98wZ|Ez#U;;SE1@A`YYO5pjgp0KvYVtkBW@5IA4tcjz+2ZYUltiS~5h8t{u-s@)N@NaMukP"
        "yX+O!2bDa5>AI3a-eUmntT=oPktv=GFH*%KZ6E_w^qm<ze%LQHGJ1AhJ+%X#c;8VnL+Y-1ofZ}cO&31DNk#LHzMMWF)E7r_sQe6+"
        "zgdqraYm<@2Dermbn_dk?5B?G-lG(p%mfa$5j%9dbd`ojKo4b?9IW8^k(@Gm=rtSp5?T1ID<!OV0^1(pY-eClgfgYfec##W(7!a3"
        "V+ScEGkJqzzsSowTkd6j5=U$(U;O>lWDflP=H!_h!|#!az6d-A4$)R<KX1&Z$0T^?^gs{t>I{c<ByBadX2)S27Y(bXzN;XWHaql?"
        "QsQyITA6{V<_^0|H&>;qBw}h_3~0%eo%zB~ceGl#epC6y1?;Gwlty=IKX46Bn%rG@h77)ILB^oHm~sH7bRbdNs>M><ZkJXZolBC*"
        "d^>p_9=q5tD7#_XX)m8HKP0`l+?HVfrUO;szF1KHDSAZ#zuzE}i@gFxTaU)w%5_Z*p}|IxkZh4-AUsmo!G(m;5jt3O>a{S)^myW("
        "3}NSBI$o{eP$RXnOtR!-7in%xPtA5(*KT<CK)$#Hjn^UZtq)$w!AH2uZ*dLSofez+ZMUPXxSTRt<S{gv$rRfvt}T2JyP(YWlN%(k"
        "Gkf9pLMAH-S=#xCUKW$CVYwSlQ)(gjzcnY>%Ru(W>`Ddx9z0gg(sBEWNcK5wq{;)j#c<a-kX}KPYh$k0inCOnnCHKSmLD~sOr4f`"
        "?DAf_TG`3n(_Tv0-PloLu4jFn7piq0hK2MWZ0ma5yD`yvf5=5^J1$t0_c>j~mHm3~Mm3&W7Kub2svVH0=B<YF(Hh%CH5No<e!Jlp"
        "g<xh8=0&BK^|Yli&7MqspNby844_wuclBgQJ7|e@-07CT{{0JE^*FHUfZg{IbnV3+78OKVTU_Q&k0p~pU9holpQE^z;?jO0$S7Rq"
        "<vj48VD1xCMwsywdha5cm-Ys-_r!bl<jL3CMUi$eDut(RO1vK<(hxW`s(qyQart+<H)EH6Pj`>#YEL(+7cqmXDwMjR=-qfkReR{z"
        "-;Vjs)pQew^9JjVJl>Som_4Fa;t%i5J8MFA>R4Z^p^jAf_Nm(C{g2mjILq@+Rau|2O2V;`e~5YOv=0JyKWs3$2iDC_r#oqPji*l4"
        "wgF}TqqvZQLJ}jGiBhTu0lFv2_y23hXegd^yx^lyUjM#ig6-MwN+<Bi=)2tq#NflHxs1`$szTAjzpd+jsBkvg<gsgRfyc=TxnaBJ"
        "KrcGmsAFAkVY#+zv9Ss7pWE+wJPs1w=6OV&02GB$L=gr%Uyr<_qkhY}g~^8?lF{v@SrMt3MEzgQd?S9tO=J-T@f`%Z?t%9)BpOy2"
        "D+AR|A`2FSkadY^z#Fj`D}zOBOAGh*#H<dIk$8tETd!NOdyMyT3A;qw#N8E26L|?fuaHYPOMgd6mVHUx-pVMtgOHB}qbZHW{rC&G"
        "`_en<IIPsQSx+O+7Iy!X0iSOY?%T1f-WTd!zaF8#m13MI&@26oBLIKwG=V%cz&SUy%fNMXMmhg~;E=&H)y42p0(nU}PoK~+B5wnl"
        "*u!P$@k_&0%IHYYbR>`OTK{SYaTj+sR_$^LDr6CZY^_@LuW+mu_)Fd1JA8Ct;NMWV{P5}G^;vXr_T%~G`G@zH|1YKbpSsHr^>S?9"
        "LD-!RT(C;)O4(%1IIO(yLZsdSP`{!HXqpg#6B75y$;t6eh^KLKEe}7G*wSJ#%5WjSjyN8atm8?!pf4v_T>WdSjwESEJMk`lb2>EY"
        ")acB_yBvy9gpC@mET&zPlWXHWNOrT`z)BIhJ6%!HTEkU1)ymq@Azv+YFut1TPqWNhD39Goi@MdSp|mb~cIt#-y)d8~@MEYN{8%IF"
        "=L_`SbR{vMwJm)ySVe2<k0IyeupJItvkaREvyNUX%qP;N9+F?W+5A6HO9KQH000080000X0RI=g%=ZWY0Hz!O03iSX0CRF<FJftC"
        "Z)0I>XJu_|VJ~E1bYU-ZcV%;JX>@2VaCyyHTW{Sq5`H(}|G?G0Nb7UtwtGS0;%s*pNxB6V-6C<@hr~UgCCZ15M5-he`!vXZ-<csL"
        "QI{MiT@<Jv5{sANa4z3`Lnle{<deU!S{J-#g;vh+!m&5JmL+#mt1p<+j&b(lho@WHNOxeRzE`#8r7%nz=7e?Zz>N~t=I3YUXXC<_"
        "yW+s4Ww+hkR;scqq`J%5^#O)TQOjFlxD)k}8IWOQ;Y10CnFh}O1%w~CDr;fS&%V2Uw*8znx|EeHNGZ#7#l(ZO4kWTl!m<s6<?M|x"
        "ut>m>dQhO0MihS6J*xGUbf9RGB&e-yT5TMxsVY!t_&#uZh`)cbS`8mY48Pj$wl%tdyAMN7G%ad$epVUXFt8#oYHlqk)gRUmg(aXS"
        "9@bBP3!kVO&E3ihJ##0ThUdL|21(CF%cXhxz?-%f8~oH(I&F^6<n;T6;;lVshjP!)N>MQzM)r8swEwzV*^{yD-x)aQ>imqsPs@+!"
        "eYRsS(<(>ZH>@Jngo3pd#6-oUVipt><sdl^UV1H#yP7w*C1=$Y^8&IPFIrZt7Ak$)l9YDC_9!SE_%36A*wK6#$bv%nW4N?}eb4Jo"
        "JT*p}G<njsHAagVZc28xd6M~U=moe0dNwb+rnSA?4HL?CM(nvQq})B@wFO&RFg1j$Hg=aLn`Fb1t0c>Ep$c7!H0fNm{k#`#gzJpz"
        "gDAHL{@Gvumg0d~eh?3(yn}#GGw+Ol4B?Q1$A1xOcWpWWlQf2T_X4x&xhS+L!<D77j;liKWqF0J8E<5T+Z*vlT8Q^(3NrBS2ou#}"
        "Y={G|Wi)>(CW0E)c48`VPdOLHKHz%a^!Z+{UR3zdzM4k*Z*tvAK8W(f{BXJDlp*EbH#=e}4MNN%ex=c!DgpS~y;k*c4>-Hm6+oQi"
        "^&Vl>CS$tp@1)!Fwylq;$1`xo6+y*@jV9l)KW#u8KtT&OyVbge(?S3j2T{D94)z&TXh0W5DV@MM7~I<e{{>HooOezarml7s?!jTw"
        "gH~<t4-!;uc_DU6-$NNl#$4_d8PhZzl!E}F$`tm8HGSs5F0bdQJ+xxXHQUt=R3D)@H=Rls0@wxIZkya}AbZ-*?Z|5L`~@@Jm%G8j"
        "@X1?;SfD$Hn=p1a%*jT;!0ykV_*1gz%BW550RSbR#HeqnKKYvE3mdCAz$7Gf0_p}~)<LN?c`0H|!T1n$HP{_cXbh8~eD3YNpehVm"
        "fccdA|E^|uLIGm|DILWRLH#2s35D~ycN(TdsIj0q?~894$OI+afFfYEgHV3WF=Pz!1(f+jLv=)+AyoyH9}?FxDt|rDv9F&%?GO)u"
        "LxKs&s1Bi+$OnwWZ{k4|9e_RBDEC6Y0~BvtszOjk6*tFAUY3`@JcJXMMkwA0uRa`(b_WlEjvy772eaS@<$#<FCe4TY5AH^ly<zh?"
        "8lh6r^nggSae~ks=}((JQqcSXtUrLv0VPhlU}+SNaZVx`Hv&d}?jV%QJc%Qe&K=6~EePFXihvxbuXA-zs?=8l77J-QAcV36<DsL$"
        "ppkk|%^kA8Dcg+Io6?e*ve`nzIb5I8_Ft2Dr0UEXX&b_UNQJ4_JgS3-xN#QwW11jYWZNwuB}-2G?CC=*h}Lpsovc$nKTyDM;&25b"
        "NC0GSJt?{xS=C6Y=H8<{?}DJ7<)V#pO6t@k>6cBGe!2TGQaVdEiwg#23JCo>L6sojI367&qXKZC6DIveE8;}{#MlT2L7f5VK&Ay9"
        "qO<vB`trwD_U0mEx5KZmR5A$5Mp3~RK_ckke|!1q4aPzE`0OS+eBBHGh3X2n|GCrXioR{wqU~l~1t!f18Wg4VaqtxWpl8@_)MT8Q"
        "lJC?L<0P5hhpQvZ7&nP0IKUHL7(yFkRj2(~jzbgkuHJAuO(6AeSwaf3MmSDHm$)z84^0{Yqfb$vkU)X3*fH4Bf|v7TDtLMovV_O%"
        "c?{ECU>0UVKt1XsUkYj06hI)*#!hcYUh*A~!vYanLxIR=hOUxVYP;P+yuC%C?_Q~kiwg+3{wo<&_hI*_g8!VCj!R!th6)nE7E&E("
        "2=uy?F@q(SVK6<gmP*7yktVt>F**gqTCc?Y_z_B%Cur@Fe);O_>rmu~55%G|v`~C>TaOMnXwzu*qk-mA$%zN|(TNiRp{H>a-V?}O"
        "M#>7FHmwU5AB0m+-#6X)wZP68%<!b=aA^ybM?UP5xRu_Lj4dP{x3rn|L}w2a5R#qpaaa`H>5uFO1N4_6eGf#4^6U$ykRh>qnzgz!"
        "z+i~A0&9%f3h@^J6VsE6-e0HF2g2k|Am<&myreB_%Me2#m-6N{tWN!h4Mdg_Yp-ATJ=w}`y2Bn_o=&s;B_4Q#%`$K<NY35BOd(h?"
        "w00BlTWesJWoYPLf`VkUxpu!sRJ7eUdVRGmWih{5wAS;~s2qNngD57`xp%T$L+qy~4TkXuY*6s#YKiKR=HB|)_RwPJICz=PJQxyd"
        "&27;4l%e_n7G^yuT^ul5x1!TBN&geqk`WifCAR>F4FDD5gn!rz`9Y^nAY*huj~dmKN_a4jQGI<k1TXGhDj%Nx@<7b?_`R|We^#v+"
        "!RTutP(`0v*=AkH(!L)p0lFgGzpIk|{>gimIvp0?o>(|D*dp|OF5#O+lK-ToO8xKz1i$w^9rB1fq3BV!>@$Z-8a>itJBLf4W_OG3"
        "@T1^wC9!^Mlp0O(k(e}Y%zg&|f(=%T7s#U*F}bpj;$e~K{!I>&s2qAGXx9x;A8bv1<3rL%H~*T7@PE{R->c2nV)TQ}4t?xp$S$9p"
        "!#m=#o%QD5t-Ro-ai&k3N)WCis7RpGk#9t~C1Ej$xUPjxXSMN}{5@bY6#Z==3cVciw*fMm^}siMN`K#*RCJ5U3voKH1OBmPd^Vp!"
        "VS<0Px&<d6E{DHg&Eb-udb28cc;s3?79V65QxT*@41BY2%lu(!xng*%dEeCRl_Xj=eQqIt#461Gg0XlNnZ2G){<?8W%y{<?5dop_"
        "KTt~p1QY-O00;m803iU+WD>Kr2LJ%o6#xJw0001Ua$_%IX=ZO@VQgn*ZEayMWMOn+FLZBfWo~JDVRUJ4ZZ2?nrC3XI+_(|Gt4jX?"
        "3tu=JijusW&1URz)jsT$4{^3^RW7oMg&{!=ED~S<(30Zlzo#1y66B1eN<|lmM0cYfUw_>crS$i+AJ*h6*P;=;CatPiORT!#Qv921"
        "!i?e8<U}=%s019nzaW-(eakG*NzZJxJ5jQ|Czk1JkSr*r$4AFUqU)8mWXH^|72EOoi&66UsL`q;{tX-!nj1Cfikt9`G@|7uh`|+T"
        "As+n!k7(bWS<R$rl<v45b$=PGx`^(Nl3USWVAs6sSWzk2i0i2E+m2T^y%N%X<V{ZA-t~}+j!XNg=hY+XaB*eFD?jp{3H|ntL0TXw"
        "@%DQ59$h&;s(C}uVP;D%D^+t{qB(wNGw$Wqb8{<l@`f0z*T+W$el(Y$19jRE`dNOKwD<y^XZ{d>8v5eJ3-V5&p>LSS0u%V~S^5We"
        "nY0fy(NNkYoCrw_gv0CTN?|1H%}&|E6(lD&{C?B2Zd)^_u_ogMD{pTVy|jat!I$^O3>@x{Mg=}Up!45iG!JyjPODg5MmUxu5VLA5"
        "(|TF5QK+mKh^-w+Rv}}6$KUWR(Mnl@@iQR}(d_p0TB+-nm!&J~$&V*RQM~e<a8I`P7Vx;`kR9%P67sr0AJOJV{#6*EBoQE@23dm("
        "#5u_86G)?L8i#$%nuL3WY`Dg{IXKlq^U5lHPbBMj;f;Ra_EzZ|(s1Z0xQ#Hdt9hp^Px=utLm2oZIe1FAWZRb4#TcvqqhnGu+*p7K"
        "c3go0m=q;g&y3Ds?K_>W9HOwM8J!Xv4Hjq)WHg5=dDtS1!QBlVY;*E6og;xmH&xc*HJZ4bDp_<8O<Rik%1eu8RJf!%B$-~3?=~b+"
        "iU;c{9?gW|<e#h^_*<=&&ge)1L?^|<QuIJK&;kf|3$S2AEzrR+!Wh9+q|Fjx<djeV5QX0df&mAC9xI;FQ<@W6meC@18WubU+t*kf"
        "aAAtq)r4LZ;7w<;RT3Q*1Du!~I(23y0mz$+4CJ7_O0+aHFr^5I;e)d85VSs14gEWGB;pMyz7>)$$sc&cQAms;To{<WFgy10&p&1E"
        "ZifojIoWDe?D(CJGh1N={oe<@N($E`mBTAYCrpP^X6eS{<z~+L1-4iTRR)rVaOwp1V+M0S=Uk>tiSui7p$908)q}C6@46CuZS#(`"
        "h96?L(Sdv{;>2R{5KJMNmx27U*cFGUKKF+PY^@h#C||^!5?wg^K98+`H;xu3Z`Ob_WP%^h{`ylQRP!f<+}Jrh$R=Km=j7A7ix`u6"
        "MOx7b7>=5Mfus-xG}W}~7VH}f_yCv@18J8Qmj*q^irG20&i0S@zZ*|?j=Vd7fB?cHw*p!M+gHMiuYg%0@5xuF>2_!tbO>B>{_^KQ"
        "ix+M+IcYTK<eE#aK?nE%s_aI=`?GUT(u~wY*P&PN9y`MCG-L-eYRRyn<<7|K0~sf;H{|?>^RvIaB-rwa+&t-%-1A%()7k+*VJ^80"
        "0Dqhpg|{&BR`4v2TOUK|Hnr%@p|b*(4AIYMW8V`jbf`_t;lBoGDyR_{marGvxC#2*aAH#*cpm>KlM(gU$j447M#TmAgWa$_ft9D%"
        "syL6rsi+O8WQ(_nPahzfF`v$nAe7`MyvYygKN-`gcsQngN9};*5#9y$EJT<SR@Wukh4nlD%dy#NPS}aX%QH7qqrI&%UJuTm<7fIy"
        "n3r2-(os>1I!@OfwmS0bKRy-d#OQ&Pdsim<(etPKYG7oL{==gd4bbL};r4$AfC6;Gf{nW(8QgfnZVw*Esuz1}W#~>0oW1_PF6aL4"
        "!tJTyx@f)Q#xNLWVDR!A)vD!mP3T~_#(5uBY<T6beF)19WCo1m9QTMWSl_qzWd*mE@rx>%Rq>JuBz_e6a!)i?`dh;M^7~+k)Nq4K"
        "+5YrG5^Xu`l>e(E<WFg|uo?(s8N@jY`4F=_T;-FsMKIIB;i#5U2jk0U0S5kyaRRR~Z4J_8=1cH2#U<ajq*grH88yzC-E4$-gCo%W"
        "<E&?6jziCSiGwZcLWw>5f?U8&6$4Tg?t?tK1hs<pa(v5g-MYpa3mt)uRju-!0the2-$pm1Nn=$Hn$`(VJIpJH(?#j#9$0p76T7I_"
        "qha^01LHLe!)9pPr-B_zpu=o=K5i&pw0gNFuq9+uPFIe!rspTi`0F2ULmZV>u*UkH_R6PWLXF{!JEcdh4{7vc;RRV=&489D!FZjX"
        "E>F&#!!aq)Brp$b2pfK(P)1*4A^!7*K8s3#V6P;+cN;(h8wza4p_mg;QVjS6q>I56Mi>FnxkpC8)9v9LF$1#4jPKzh%vmyUR^-j0"
        "H3{;Cf-%!kYO$F<zZz|a8tcjLD$Z+Ozk<5p|7KcI2|MnX{tFPcYWEMLdh|A|^M{w+ga^k#Ba!zi4j#{lX$&?`2T#m*N`WIp@|0yQ"
        "tC?k+Z}S06qv7y*zTR6l@`Xao-l(wLxRweFP6dr6>pC&3RBIkR^hxIRV*+7>mc4?i+`9?V#_uoxUnlr*#Cfr!`~IbHpzA$wJ(hu)"
        "!U(dZbg0oxIBEC2NPYEY;!{QU#P?ATI{PmVsqKZa|1wc~n5N)=0Z>Z=1QY-O00;m803iUS1~;4O0002n0RR9e0001Ua$_%IX=ZO@"
        "VQgn*ZEayMWp-g~bzyXAZ*DJNUukY>bYEXCaCwc7%Sr<=6o&T$^A3ls6r?X8F03A<(2g>7A&7)bdZq?*<4h`jd=p0t))AUj$bZTA"
        "3n5zSiOFk%K?f#aih+~yTj&GB-1I>!dr!`(OX#xi%_)8d+LPWT6Fis!d`MtiwB*Pq+RTIybyZhA2M66)uq_2M`cQVKO6=<@b=Xiz"
        "<`3blt{@+Le#CaM!sbb?9&T~Dz8K$Iyu%io8<g|wH7*wo{wR@+Pr1rP#byXzCe>N`CFV`a@JSxpM`BZk96aY`jFie+Dd7rU#+3L&"
        "D!%)~1nWVYrahAX;hlCJ6BpNOU3~&jO9KQH000080000X09?f29bpFm0Q46C04M+e0CRF<FJftCZ)0I>XJu_|VJ~HNVQh6_bZKvH"
        "FJ*RNY+qw<WMwXJdDU3^Z{s!+{yjkchk=SBcHzfn;TApTT+|Qk)_`BIz1ss0g+NQRO+*r1l5*na^1t7ZlJ#_Uvx^?uOAy2sheOUc"
        "Gv7zbvaBUNaG^j(hl9_k2Lra{F7e;fARr)JE~o_J^Kd7bM?um95kyh{1d>O>#R-CLsi2`tW$XIt`f9yr5(4f;J_V2C<d7?%kJOD6"
        "g<qn8g!Ym64VVrViw5Y=3qO(#-~v4zxMYfplm1AU07c}zcDsl+K9hS~?YtvF!R4W%j|zC`pYF^z!F?ZXTpr*qndG5FN54l5jRh4^"
        "7_xA8eYGKy`Yc2*4@~Vr?J+D8;BE;nM^yv%MBpgy2_GXU;jRHB^g!`@8nzYZK5QayM-^_&vMe3+R`9^gcYPuydS-z^HGPzrsM7)6"
        "ok!;f#dt{k)Wasmey6c-VL)4!+S>nG*vs+Mo~@l(e`>Gvoa?Kfa|@V%Fi3)18JxP;SAaj3PIHo_1i?F)@sL7534#=b<VcuLHkdQa"
        "gU;BI3bxhtU^YgQY`2nlwNuP7^_>ur>w;7wez?~OrkGUcfV;WvrZN6|>vN)hgPvJ8@vu1@USsF8BXx1L4eo#M9Dn$<wS1sXZ0R#a"
        "Uu%1g_tW;S;8N<X=ny~S+5qN6B5#K{Wmfw4y*-`wJI-`IvM0{4zZ~JHwO8sOyX`h%tsUn(pSC=DY@4cRd)$?6oxygDze|MYbTBEa"
        "U+sZC8~)-<?Ec7J4(26i4-XG54A(@9wH35+&Tu(jt%kU^{jooBmiA(vpRvR4h&INS1mq7MHl@pS3AJ5U$8Rd(8<=DJW#y9VPZqj<"
        "<WcVq>nj~UHy?yiu%RNh)OfUK?f&)E(tc+zF$~)o^jG#|JhN9_@R+MKbd6GVYbc+R3O9V!>@@ASgXVZ`bxWqT*zA&Nqf%eJPN(&z"
        "k;`DOlBU>Qy-{Ah^ZM00FRlLl+t;uD9GB5IFeD)l8IlJw-HstxNYKb*h`1rSm`oz1ZY;!@&5-tnlQD8wj#|tTD}?nj-FS2h&X$XS"
        "D90qahGh^{d#n!4pEM^V^dqQ>L<kmU*aEf%GYL>83<XhNQrKEgv8&eY_Swo-I|vpkCnI!6X_89tM1&kfH}aLt7fHN_#zmIW^_^EZ"
        "Qv~>DCFb8H|Cx`Xq%>SdoVd}P--CAqw~7WPJiYgfUKe_wIVm&{r<!`ET(XfA%>t?fb0uzB9(-~Ff=IOctuwK%s4A@ATML8>VMiI&"
        "f|T_Rj_)a|<er3@$z>#r9vehji8I^->PPw<?&&FRx|FILhKOU0D;HxcjWGW3z&xy`9{l6S_`AtZ&FWe3T|8D#D4T4khRhmICLdPo"
        "Wq&>XXajQ1(q2JrHebWk?yo*9?NQy7L_XfQBCOSi#bPqHN69QJzu?vacrt4|ci}0^@v~Jg><b{%y@TM2yWGdq31aTCEe_lvc36j|"
        "sB)(jsUP5ZLuU|)iwmYst&;0X$mg9s-Awg<Zo2<``{OhG2cB-yQxDX%+WmZ6*Mo23L46L&KfnE6Kc5T-aQp#k-_}dZHuKBIBf2tl"
        "@C`$*NV{->MO&zafvnZL85PAIv+u^2^<;YGy!T?aYyYj1#MjTESQSqfCTEggKZknBtXzim=bZO6@Jo<f3JkZBqk+qz<+!AUN4wfh"
        "8s<&?QizrCDh_n0W^Nqj7tJllkABVwys`@8^Lqy-=xK<a2HKM@sg$FzaVo-M`lraf@k#vuh=R?<p>lB{`Kyazj^v`?QzMBUBfF4G"
        "#oUaCQ|i>5*)1I`V}`Gs9}TtQ(%e64&c*fQ-G#HGR=?cpNbD}^0#WY1VJL{Cfkb%Nl7{N(-=B=3hKI=hcnNJ7&Zmosy&lhJFdFyY"
        "&*rQ3cv%0Nv2~#*E4+<I>ZZvqb`d5V$vDT`+2BViT7|)(&a`ei!}<2q(D<e9{p-oRSkqs|<<;U(=U*}})c;>V=*3XKmes34f32qi"
        "9$!xIS^ec1{n9P4)Mi_&Sf3Y35M5O-hcBxGebk*7;NiBQ4oSSu=H!@s<t3sl7Qg)|^sjF<TlT2mTOi=gBT5ztRa-sqW_t5HGpmZc"
        "Q%d%x_a#KwbJ~ZSUZ=(*sk^XpNov7IePM~G6l<&T(#wVTj9nChT3H5~Mg{-@brDapqxFRM`-R&>eE-ib5qN{RD_lTOhcrFPXoCq="
        "kusQw^F2Z5N|psB3hzVy5ITwL9%`nlm6M3ro97iok}5+1Df-E=lpqYzGQinFaDl~`0M8?3blSX)V~<TUHO>*%qBJ^OEBdXch5bJk"
        "xVQ;#-*ifSl)Hnxn}1k>=I17MAEKcZaJ+7g?D&h;#1-^sBUt}B#{o#Gydr^9oqfzzCV=sY!o0@x5(yOXOWVr(Ra0&O5%;nrfL|?i"
        "MN0egR4+r*6lKch7ke#NLFc=3uR<<{RWnBoczUjXRv$krBguNNmynC(|ALoKOUUOvy(A$o$ViO&e*sWS0|XQR000O8001EXH>CH@"
        "8WaEk*GB*VBme*ab8=%ZVrgb?V_|G(Wo>O?FJ*RNY;|FDX>V>XXLD^hYc6nkwLEKc+c=WnQ<eX~<9#?Pp%uw*Yoghj$~jxPojh)w"
        "*}5u6#iB?^;+Y~feE1QU|NC_}0HQ%sw&L7+Yd0d%Xf*oW4RBr8{gPyvh@HP*d_4V!vnj+TEW#?uGbhYqrx2+KOR-m0cd2m7C@+$1"
        "={u!_63(K?*AC7}3g_be*P-jWFJHWTk*qg)Q8{6;1h`T(FIHi>N|Sl>{J&+MH4g>A5xyu-QxPEzrK85)<#kqxLIG^T3eo7nFYt(~"
        "{&D`#4;LRke{!addoUi2qm%e(bh3XqJ{9|?^Rv<DG(0{&IG)GibUz%8j!)0_BOv$U^PhjbI}hG}`2G9>rry7N;lQ6;6=9OO6UTGM"
        "CyRr_lQ=pWAD)~<N2B;qEas;NXGdr8d=4*;_s>qkd3btyERK#24~5tdLlGa0_76|T2ZzT;uJ4>2o*WE(jaifxAwD=c7{v$U{l&33"
        "I6NF39i4@dI6OT)7>lFv==eB{;6En^Comq)MRayHkD~dRIGi6u2WRp45D<?K#%GB5;pK}DpDw=s@#nj*k^`Pv>>^LEg;TABvj#SS"
        "fb)4O)=pW8jWfTae}~RFi25dh;eUreX94`rZ$Kwt949W4EKHrOUW+1$KrPA}Q4E1`17!Nk7qM74;uhEyRe{o%KmvJDzTTt)B{`TV"
        "dR+HoSVb#Btf2p+ge8iylB>(9q<t`b9Rk!kzYf!1oGfpM5b#1O8zNm3#wkD!OA!{)$}3#L<$c@x+vRBQ?A!30KYKj@eld#V+(J}!"
        "kx3N8Ws%n#Z#)>5RgrAG!4OnC1(|rR4|832;3NyRnn+9Ge9ALnm~aiGEfNt2WnLGN2#Wls^x~udUCxU;u$oPpRGJaub_4Gbcoq8Q"
        "DMCeut#GBhq=_>*FAl2^#fwUBkPn<Ujz?h5YsVd;U8U{-=?~*zeg_)osr|wRlobwP<^7{$4>1mgD{&hqOVCVj;QZUvB2Q6F+zJ!e"
        "_CLe47UxBg7oNK;*QZx{$A~#tCnd3H;sPyl&x+N+4<P*#sT-zw7?)m}WMTjcS2*~~Niy2skY?c<AR8&hpR<PkRAe6FvI!;5?M^Cz"
        "KSHZi3tYwscp(5Z;Aoh9nTCJfRC}-$0(bHS_^EYWg<yBirk<yuzW;Uc`P1*eNQJp0d5F@P3*durz~)!rWQ}$!FIN{S3at=-)gUD#"
        "iPS+a4~%W8L<?ZJzJlK#IAoDk<@9S&3)H)$tb+WC9@Pw#2C>3-JX=eH0ZNFg<}j(m8id679s_mYFerpf8GU^UYG#30*~B5yW#h%a"
        "-7L(W6XhA8t%Z3DoCL}i?y9Ob<>bwqBD@(c!TZ(ox)cS%1BZs-Dc<B8k%h^dWf8^@1EW=*r+er|-`wk1^B@NUIT+9&s_>h8qUAyD"
        ")GSVJ)|TZE9&sfP{`H+IS&O``rpKd!6PBQe<pu(wXmLWh0X<5F?wtms(Fp!Kwi%1|LPqZV>S&OW15*A8NAQ@1#raK!<}KYpUIb<G"
        "U2tJ@N#7KpLdr!E0mj_c;T4EjK+*x80)r;W8#siKS^a(mLSN;e&m(4kUuv+UFb+46GLSms<nk8`esN8Sg_Y<itpW?oK{isgAR?ur"
        "WJb7<>p&2y&<MPd@0<>hiJp9q@d_Djh#NR><($DTUI_t%bq$-Gi{QH`a%7xHma9s0PTYnVyU>NY-?DERSW%v2p26+@H)<2?XyTwd"
        "0~#*fztk&thIaQY`;DTj0)TK?&mmU6eLH;hw_ksy5a-KzG77shCN*RvjN$IhCNC#MxpK}r#$=ni0^uz4G)9L%^Wne7x#WvQDJowh"
        "6BN`Tn=~@!9Me1Sb!ITdiNmF+JhQujZ(b_WZ5tr5l}T1sVFvNsAcaUzVnhJ_R)>H6o=ud<<+L5m0b&fyUj`aG3bspQAr=BWGics)"
        "TQ(*CvG--BB_*XJI}(*-kU=GJ(9G{4DT%=kqGc51E@|QfWi+{$BAkp4;|J#3q}+$$2Ao;!*@WENkEE}ft3eV&K5Ulrdti$^6#(R7"
        "=I8bS%wZR+by!>>6w%rX)~L+HXx0|e3h65VlBL#tz!WlWDW`9Y^tCcl6jJk&l1A4ZbaA}4thaAmQ`<YYG3@r9AIu|L6%A91>3+Ky"
        "9BoSzqh}9xg2YYTZwC~;?;%XgW8n>~=wckbdV^)1HY$a_g^)5F0<$vPEoh-`S&Cu<C2QqfqXj&U+^~ld)VN-6N;wbA9Wt=sO5EwZ"
        "Nfp3(QBGBDFu`<N?^AihezIif5i*bTR3^ic7zqBr)4tFqM|7qqpG;aWpZiiL=>*<GL_4QJ<q6(F$^!30ClfcS{#PrQq|xnh={{~@"
        "k=$ZR0K`~f+AP|et95rPtKotFjEPjB^O=%e^uN9pkQ8gW88AzEG2mz6TKKX~!5Htk%A?67&R1N3VB@)~BnCqX$ddFhJ`6bHm5d<w"
        ")z}9r;X!}V@{q0y=uG>763EfB67rcKETbfuz7N5L45Z1sJS0=tL}1WB(EF)dSBt$<w`2b6FiRG|a?|OcwE+|4uIM`F$s&)!bSa?r"
        "R<#cL+~9i5eT%DwBd#Gjl9SXaaSs$FQvh-gCNzO+uG9ueJAjwIL(kjFiwBN#TO@Ev@G!y^fCgf*&e5M!x)&Jb1*->e^UUXHr-%qI"
        "DJ9}=a>iLNwPMC+HycKoPqc$;w1G5%P~b1pwYIADD45L#NM@@%%2QmWrAWI5FBeo2s?wl>g7XTL27pPw+F+m#QJtc@+XhiV@l$Ra"
        "jD^%m3e1HN@m)JgxoZI#SA+mM=PATefKV!II0j)9)sPW&=TXKegH#DQO+ZZSp+j@-!8U?Rm&FJW>ePxE8cJD?5ud9^NHDHt)69A`"
        "6#5Mes?&XKxLrGKccbJ?6Evuvk`aUXz*mSmN^2-UQBRb2&`~bWt8HBY<^@<c#-k}c`=ek>+w0J+;{#<46}BOykxP5-ns<X*nKni)"
        "fWT7I62?Ol*P0oW-77~$wdrHN$|09-gxOAOt2-4T5K(IhM#e6+o1P8YSBWL4!ks(VLVeq7(&|m$8c4Rm$<@iNlDFHvy#+OfoUHY#"
        "tka5Wv2MiM+5g~vxW37Jwmp@zENPNErCLoK<ZQoG{&IGyhi5yMF$iziPF=ia1;#7Yc(if0+mQAs?@4xuf6O4P=B15Obu!J%Szob5"
        "q&^;Ta6>bT%Y3@USrf-*s9)HkfXdBd@B@+s4)B+|3ZTfpgO+HUYE|8hhE1oAttHei3A4La4%8I!ZOymA@`C1~-;mtUjz+4os#-8d"
        "3=2RPY_P|R&t;z0vSH98ZpK`eHtom=wUUlOXn;3G4zY`yAt4}B)YTWIZI<u%UP%bwVev<Y0X%`zl|v^bj}v)lrb+%Gl~PGe<F}Y#"
        "s0CthnSmeCCu>$qTc02eH>ev-yLdq2QClM!e}Ea3f^wGW7TX4jk2RJD7|4h$^UkJ<BLseNIHg4<gdsB+ZfwMkcL^-XEtElKV!+na"
        "`<mauR*~2T<Jh|tPWl{;?4-3Azs0NQz9*NQB}g13T6VOTI=jm98)}BdHF(S@gbd11%q@n{UYez5l5hz?=Qt=Wbwuo05wC28JWM)#"
        "-Zaac@5!cRN|<y;m8=i9F%A)J<(%4vK)Dd{<kQT2wzsCIDUT8v?+udv&K|5>)zl2u4MxU211eIEOJyN675(Aoe<&vlKepmuicF~5"
        "NuUTV?46SdbHWO9TML`9b`PxEqBXF+-PXr+%S>%L>eQcWt=aJY>C@{itns=uxw^Bef>0L%wyIWEg6)E+yb2r{rtL19GMyUjHC=ku"
        "ScKe6&1ezaWp+m+@(pZ1q;#L6HMp#0wF=T?om6erkYYx=yX4AYO_@2br_R{SQ}B{??LlR%JGCU0@=lZ#;0@g|X&63HtZFqXd9&jg"
        ")h1N96(-;^uBkQ<8`g}UG>p-2tiC{lx(5yt%&o`~)WtqRRC)5r0#N*;J}vz+<*v>gq-AUh&}qn0Am=SNBBEZGDn^mHRH@^SSD1a="
        "25KOv@+*;*)BVxmsddaohuQjG2_SYLwY&SJ>EJmuFZqtaA}j<IvG6uW&#Ug16m^#2Z4!3QhJIk|yq)$?fN#eK<I&0f4$)H^qDC>V"
        "W6*V0nr%%sb$)5q_?b*|_snId91^yJp2uth1*tmEg<X27mp^zkq{E5DdK4n}Dw2u{BDe4qpzeNCPN9!PX1)`~aey=?P?#%4qR`qs"
        "qy_|-8g=EqBjtKh4hpzP^BYdEQc#xi=nk1W8U3@A*QCmN1L+8;spg*A%1~)&E95<i7yGp8Ft%Ze_DmVzrz}Ak9y-AEE{IlP1rZCP"
        "51xvewJz_Q<|p1%Ks8gh6{wAgggkI}qa`<Ox5H2*%yG-9@@RvL*lG@1NK8__NNyWN(7Vuzp!V0R0<u*#u&iF80|S%4E=MyoN{x1&"
        "mvXk9Wp2|31mEP~q3&AUH6TwPVKDzF*`bRo)h2^=Q?+ICmx^4OF3N|-JOaGAqZ#zd*NwK@AQLy-gR+qx|H?)w<OFK>8s>0xb__7g"
        "4gXzHE-z!M8%B~u<&U#m9>s%usj-eg4kJ=Xd_+WE$~a&`!ux7Nx{;-JDKAW@8%flYY))$$K5WC2WGZhN8r9RNWLsG6Ro`vG?imVT"
        "h^$9Nz<+lsn{X28TXVllYd{m=0xSai>8HpzY#^zB+7j}>_+oMqRwno;+9uTYqeYExxpaQhkeObmbf0j`Z4?|x>qP!mGW%E_(4(Z?"
        "SFBlLerkI{?`BE2wLCX5PQU#|Zl?9JWn1=R4{WbJZ5!}=cLnUj{ph_7wL4-wpt4!s2ZDbu_r^0y6~?|Z9v%D6(WqU?NSp{e0bnzB"
        "mk6-gzOx;Z8F%aV&JmAkOH8C*(IICGiWO#)#@{M=0cB8uT_u}-7v4tI<mXsZb_Y$JF=2dU@A;`sk9&rM(E~qL^6QpGvZO1@DLNYm"
        "3@N(W<VnW;UszW;cmnW@H*~40e4}rvQ6EFP$0!nI*>Z)_LE$w4s2W1LcBgz6n0QI|F_gEGZ>zkBR{CAn2|~PHB+JI_LAGZEv~3QV"
        "NI3)>pd)YW2*~Mer$hS>@M7Jy2-=245apr@)Wx_m02dN8PT2%@6@b)kF{54SV<bp!i1|Zx_kiwlwhoIL-^~D<O9^zu!7=~bbp_M$"
        "pl<@2+X^xW`uv!S9iS0m>J~;kFwvbXn+rrjn&-hb3H^A8e0De@lG<_Or=w9X*4=}VmbUGJowrcOsVj6;&=HJwv(zSdRzb%ySG%U;"
        "y-(5rhBWxsMMSA38QWHW?2tu{qykN+k7d91FlpCow}RKXpvDnIU68rFSFW-mR0lkY*XO|GB(Qe4hIC)vKHTC;7?UJ}EJBUbRRfwI"
        "W}sJNlrv&rc5S7|x^&TV86Y~Vf5M~nCK}sT?MRIL_%N#Da9D^ED|-<_YAM!vaTj2WYfksLpObfscD`z_!Y(CABl(K%hjWy4Gu*xH"
        "yEB0I!&i`Kq`V;I2Pb_sjm?i9&2MF1(v`M4r#<fC@#L&W*Lx;BX;`~vbzLLk`CQweFo|jo@!K~-w%ir5QAjDsdmuk>fosr<Ws%n9"
        "s(p5BRK;$cl8oqv16%gqZNRD1sDAr)8&isQ@G$TlT9RufL`RHY_9Zcs+P(Yy@yqY$U(f&3>XA}s(pr!WVvIyP;#{0;<1U~0WCrUR"
        "KOy@d>VbpUs)tSDM-|ewv`&L>Q$^R(G!4B?{jhgK-JQGu#U<P6xv}K3(>rrlO<ktoIhZvEeN>pD^Ttm$=z1*7V!jM_tejW_=XG(Q"
        "%PX#*M4`cK4BAOX;JQ#*8jtLR3;ZI3JP&zqpK=4rYC-yroM@>Or2F_mOwT1ZqKJ+G3QtlO1S%yDf=*QeYh)$f9RdiLqyRnpeP`Bk"
        "Q`>vTz#5C|1d1(GLTlgcJ7!jz+kZn$TG@{G!^~$fN!H}cboPNMpZ?Z%_?v3-e4!C97G+c<8`Ps|5X5-|%uy+CJ;xI|0vl3?uAa-M"
        "5@%IYm%hKuV*DmV{8cBQklaP63>t92L_#2Ed#)OBZ9oFiEhx|3+mlz7_L{qCuugTi5mPKR9$m77Nt(j*=gXU;YC7dG(JBWams9zj"
        "59WW1uuiM#=rK%8Z};}->p#F5k_}I>^o1Z$0Wt7$CDP5*eNP_<ZsZ5p{L?^b7t&5ty{g;KT6lQLp@;SK-sM1@&FaeMX?AU$-&{g?"
        "CW>9<^;G?Q{y$&N|M>9n{L|Oq({CToFLYg{6Q3Ri=^@P{{rnSb+KPfjE~8Oxntu?diYj|77ybzXv&WD<HCtuT&z#+)k9#r%K7SXx"
        "YSWW!kl~xV0d4kl^0}jAwqwEG{Zm*Vy6!<rvZoT2Cy2pBz&LUG2eI+5y(w2vo}5nhwRg793JHvSRx3O1`{<c~+n;2xgq8JO?5|yV"
        "C;GhJ(I34?Ncwz|7ligzUi@R>)E>aUp0CcxJvYWJGvVIrlI?7unXJDvFeGk+{Lx%j|2^2R7zcE#C+(AL<J*`ByNy$g!8*p=(y!}n"
        "J?zq%lkv%F%F!~`MXMd!bFB;R%$o7fzSQ^Z`@MQajq6`=y>~%33Yseiz2#=-QZ%nb`!DwGdOg)l3G{~}ox?Gjuj{_uZko0y{!AUM"
        "G4vEZGx~oLFr`mF2|v+@X;1%uNY{;ywu|#Cs6XU25+8oxwGzT>eGx&IQS`N*hk;7}VI-i!BM8Xgg8++xAkYRcOM{m${tr+~0|XQR"
        "000O8001EX`~+qRt^oi5`T_s|CIA2cb8=%ZVrgb?V_|G(Wo>O?FJ*RNY;|FDX>V>XXkl`0Wpi^baCvP~OKTf35WXAyABH_O;BM0E"
        "9tvAxlR!v9oVG_ZmS$~XJ<N=hK7M^iyLFt>=pYzL^Swq*(=4w(b$})p^uTmP$FO6xF7Cj^Ogx~W7S=>ylab5ixBl*H|9ty!-v-+@"
        "O*5a(X9K4IN6dTgw&21safU}NU+=bVNCy3mLcrFfm{}R+^mO@xzF>AKK2be*uSpCJL-O|2vm(re=|1HrLi=mUoVe1>XU3xltiLb6"
        "P&E4+<VYgjd<MEK4L}nceVC5w67s_Wj!f2>DnmBaOgDO1fW?eLdBgPLL~lEg%nRTWH_nd5gmeKPKmX^YJF{q5t5QY7butIyiV%rA"
        "SoODeo9BnkW4~T6pzfy%FskD?Fe*ziV>4B&(pIY{<m*Y{r2<00=mru^Hx$V_Qj^$Lhh#EY(1!;ZVYfA&fPBzYVhS!|PTZbBdu?Ke"
        "-u+goi8Y@3A1s&@j>t|@RIsflZ~`%9K#*dLW<3;t3Rz=l{|Vq+Fgl@k$M|M2W0(HWW&zFL(Du_nfqs%Hm0&c(B*kad!J#U~=|a=0"
        "%6>SVZ2N}$Vm|u=P)h>@6aWAK2mk;8Apqv;?GB_1003|$001li0047xV=rQ9W^ZF*Y-eR{ZDB8Ec42IFVRUJ4ZZB?ObZK^FUvPPJ"
        "Xm4&VaCx;F?{nL>^?UB}KcMV~lB>)(Tf5t-RXW#}Yu#eWXQ|zCZjJ|&ki|1avH)mX_42>(Jpd#@3RIGAQ!Ww-JiPbte&fNO=gp~1"
        "9~imX$xV@yTb9S0E#>z_#05|DyAfeeWtPUNByo{2cw=!T(;^>{b^62-lCZqkra6@b7vaknFJGJxQQeljh?x+t9$5ZBQbF=U5=sQD"
        "<*D2eLG$FccnZmi2`LD#a#G~kjy!HyPB{Cmf?<{bI?wYEXSyv54j6=No(s`Da@IVyq?;khw&gm_T9~Rzlb0{nyx5Y2N*ZTW2qs8F"
        "BS~X9YNrfnN#!O>Z)>nCcty_C+-<rpVw&Bt?UsgdkxNcv*<ie8Wg$|K&u*F1Qfvz8T88x0TJ|LCl`+kWJOz+Qn3d6nip}71;4$7<"
        "TPe$I7pcKW77Y$msp2rI@^qcDBuditdIU;Yvxoz;xI}I4<%_>JV(@|H?<}9Jc*TO3FVq;(#<Qa0{+fv@ljD~!2z=69NSeniN|P}W"
        "lIu|zla;UrG-t>qFfk^#MUm-=b(P0xW3cfO$ml8ZZ!#}(rXe_^qR164G9hRrnnF@k(lN?7w@2|`!q&>hSyU?1*K+dR8yFc(MuKC9"
        "P7O%O8QM>3m@}gi>^TxKPfKYJ|EXT2IncgEZ!=~Cf&Z}>yrH!Lv7vwd_OCV!&9dS#LI;f+2Q^ZgkaL;|Rx^@r8HgwXPb~5TB)iTE"
        "Dgo#($9-~!sRzbIy&Izf0LLF#5RjucgGR9KQzPF`$g!DKN~Eb^<QJM%Y|44TeXm}ytw3J4pc)i1Aoqd>=YUW6^;<x`m<-G85OMYZ"
        "FBC=BMICKZu|?ykxsZHkA<|)RGURRkVYl;Jg9(CK12HQl@mt<TWU(|4ZGa0`T3FMkG6fs~L@vS*$lfHA70>rH<XiSVBHq=>>hvNy"
        "KbgIqo_Ql<FZB<o-B)OXoAI~D$M0KBsF?0DypPjwA$r~h+$>iotJ!I^n7@4o%wV_~X>%iWu4*WeBo*i`N$9m0y9`>M7s6K3(hojc"
        "KC}7x^+`0nzFu7Kaa7cq6A2MRs5X=@$xlU==r}hbn>2xBsQ?9m1V~~~8_Q)1f+jfwM<ehC-*Usq7DSqXWXej08rfqiH%&4NR}5o3"
        "<-0RbevE+#EeULK;+ec~g!X*oDZ9w&mf71b3iNr*VLna;A!nk<9+)3kc3p6Sq<{%+;~~C@aaVVkDgc$QP`>Kr5MQy1Pl1z!r6&Xm"
        "Qvqu;r3`Q(g)`{U`>5+8qKJu@mQB>Kc?E7bNIBfzCn@*!i<oGgvZoY`rMRzi+}_$pC|+1}y8{*Kt*Lws*q~|xefeG`*GGTz0&A^6"
        "1?LRou=nHad~tg6Hk$qdU+1&=Y<V#~i#+nRDoH~8`o<5!4STZID~p2FM!>!o8U03P(*<{;fFUm|ceV_^B>!NnROYpYiY+5p=z+ae"
        "Ii%BggPug_8ZD6Kgi#I!8Hhxp0cdl?O@=l;h-4D&Io#4T_Yrua!bX5K-y?^R2%%9FWKGiyipf`Cv{?OBAVvX^CCxH5W#R0`Rz1Qn"
        "<3R0)x)aQ|T-w^4iJHq@F6omp3s*oq<n;T7_Q_#jmV++RgfuE4b+Sj*v>~)h!g{R*8KFB0*zU7e4|yM2FcfnGBAD#^YmOy4s0)Sl"
        "NV9t%h1Yi>@VHdeTk~<rNG+JDs?p_eUGBoFgsPA^<fX7fhzhAdNCF(GSM{<sZ^$1K4fQ^dn+9eAUefG)AOC*nIIxj(sF4E|XB9QD"
        ";414;aBQ`e@c((6NXWg#c2F1DKO1&0(Z_Iqb3nV|G#5Z?%XnX7SlK>+U}SU(>Dd=7gW?`paj6z(!#kEUp2m?{r34u`oIym$GY0zs"
        "k#uztVim6yav~w~-d56u7C()Wa7jod7V5(ys#mz^h3J6B4Ca{GK82uK6N3A~?z*P#&<|?cAKt1W+QY9gx0dA+?z)z*Lr3^g*zVcx"
        "vET_h|INL!wEM=)nmIsqA5wmYb+Csa*?SK$=*Ep!2?N+Rt9@8E(TheziDiPbr-j<B^mT9_5bAsPunD(_CoaleltJ>sOg&&lHAl*J"
        "he7az!kyGY_tRw3=0-RM$M=y+G_aFcJ}J1j1x@@Q7#&6nG7ys+6zcnrK-My!+#G^DL{9}R_q0RGMwGb74imD81&K+NRjyMdK11z>"
        "Cm68UyGHiH?*x?~)Sj=hg{|HV<vAQR_I=)~`z<^1Yu`uL9*l;%pvwj0(mx)NXH0^sp!<S<2=>-^7_?m4gQ~W;<8&L0_t$8VVD1J6"
        "4P2-P+`!-(1|G^2b&>@a(j%cN%de{(BhN8Wxm4D3k5nwH7!O_RD)Wa}QS7MFbh#4{>!;vG{%glxEmaz>_hF#GDN3w+?y}+*yw2ap"
        "v-a>H<j2Lfgj&>pw#)!zJR)CxuTLCDO;hhkZSu`mFo#RO3PwcJaUm~x3LgJl7qEM3vx{7P?9t^hHn4nSI8ak#Welb_scU1qe7sEa"
        "bPIJT%O6r+<Xa&19mt(roWJ@hl^50RtCOo)=a_`aHQz8Kl#r`;tBb`PPOleM0jW-;$`PLNV0+==k@DSgHNCuEELQmT&)NKJh9#VL"
        "y0~0kO;0_22ELg7OU=JMi{8$Dygs>p7byf*{e|63Kn#9A*5OeHVYV~T5~Fb;LbIO`Gd^feg=Dsaf}&&`U-O#FE^2qhak03ZP8#p&"
        "1&g1T)9b72#W@_^2Vqv1SHpADLz9<RAgIA<E&lled05U?(~0-mbAnu)EH9SR>Dk0Pb_2{7)A{M*Y&QR?Hj|Ol&OnYs?Wr!>8Vb)%"
        "EEI5-i8ns?eoemqHH0t!hq3w%e(**^824+<2B`@iu9N8}A1xHl&puw%9GISiVKw}Csl0!z!jD;xXNk70#H*pye4vKd>tG(pdEU{i"
        "ZwB4z_~5k3JY_~D2D|TcoT^{Lt`;HN4<Q>$+2Dv1_&+ck(-Kt7qN0*zC3WM#4T*8^;Jj|%O<o^60D`_f>z3djYy;Mx|F+#V&<m_T"
        "G?o>`PDgE*x*uQn-S!-+2GeY*>m`$S9R^q<Z=1W}9WN@ti;MIuP@-H4?3DR?P}mRYcZ8hXr9bS$Ziuct-7p8t#B|zSQ)|z}0dQHg"
        "CQ>)5$8Dr|?#lrR10%|3{re$wab*c=KXx=l*GI#~&3<6e{;zDwG;v&>7(XTxOgfIKVF&#*=RD+?{c^<xf({@i9%m49V&(#=UAuSx"
        "(SzJ?UJZKnCw8#9e!QApP8UC~8Xq<d{kn^;8rFuxj$;^ROsx6bF53Sfmf)SA$=#~_w&H^-&hd2J^i|Wm2Gh~Ts!G*;BQ{>IPL@j#"
        "BaXq$KoG>k3lA%E!nAj(nnd$Lo)=Y~sQr6g3cxbP8&CI`1|WK(Y8kP`(V?umyI;6l@_oVY-<s8T0g=@QM~rZgqWK+5Z0&yjm5ggM"
        "{Yr7)n{cQva$lXfV6c`*Pb3;jHyL{9yuP1xlpY#q^lP&K)BAo6rwg%R$r~@YIr`@K_;}n8G##ZmU;^u^xZ1)AD*c^BxxSy!YlUxq"
        "$0SRIyy=Vey}BXA3%>SNG-J8{teSzAlskPxgY$#B%S3Q95FEU}Oe*|C4LCs9W?+#JwHZ)}7Ng5!8@rxuybrc)ro!HZJXXL;zw74w"
        "Y_V<dsLS<_<H2;V%;FF(W3<oh_Pu5k-niY6r901}rC9+${a{1s-NKZ97(jrN5eLyWfaH}WMo?<|qN0=8`@|Yow%yC-9^?hV_=7g7"
        "KS;%icwJ>#6A!vVB5&##y3b`-H1(fn>}sx*D#Om-ZD_sODzlmH1Ubj3lXo2*m;i#$uHaMi(MA#Tw8u)Xr-An}8WLax#BY}P&kdgb"
        ">JQfP4>o^h9@x9le=TIpJ8;`@9Z}%0{~u6G0|XQR000O8001EXwiE8C*8u<k76bqQDgXcgb8=%ZVrgb?V_|G(Wo>O?FJ*RNY;|FD"
        "X>V>XaB^j2X=8M0Z*FrgaCv=F&2HN;489xiI}keASYix24IZ-7QeeZ-73;7ay26kpI%O+M1|=^Ay1x4?J8hgSEl(o({ZJBx5bL+^"
        "N^ng>ill+dU+ZO!R^7mt%WvmjVM~nAIBsS502cos16gHyjj(Y>gXBPsRKXWQh%`wPYCFdPZoRW<(y(hmIb%=-a@H5JR<J{S4#=`L"
        "xI=X^+!(5#x)(zl<;Gr3jZ@NWux(|boDCcupP=8dlf)N{nnssMOB0O^Y?%U9ev9Lcs_q%$v)Q?P9_8v->UuHGRe}!;f5)M6Sbl_a"
        "XR%BZ7=qo_!wRmhr_>|bA@vxte5wqO2Ya{S^<eHjVPa4-YZPRCkXB(uI#QzS3_BW2@(4er>2b~MIBUeGzB8m`K$y`>csQ;5DDu5x"
        "G$^Fj*=g(y882MOu0yM{!-?Xb7rJjdf7FQa@P8?Wn3;(3g%?=7z-^sIQ(4(!<+zGJbttj#zpgMa)e&_ponSqlsG;>eV<o*JTAfMb"
        "kL_l!w>t*O8w|su#Ffh8<Ohwxp3#P)z0<_ml=#&T?r{NlPd-%cZcGnZ)4m7151IIFMRDuMX7Rsrcmrc8ZrQl8I7|sb<b$47PLtiN"
        "L?4>v`?wVU08mQ<1QY-O00;m803iSr<!oF)0000(0000c0001Ua$_%IX=ZO@VQgn*ZEayMX>Mj^a%FB~WiMY}X>MtBUtcb8c~eqS"
        "@(j^UEKe*-RVXORPbp0<$;{7F$jnPiElSNxPE{yM%}vbAD^^GZ36`X$7bSufB<18MXX_~`De-av08mQ<1QY-O00;m803iS>kw?nl"
        "2LJ%%8vp<(0001Ua$_%IX=ZO@VQgn*ZEayMX>Mj^a%FB~WiMxCZe?;|bZKvHE^vA6S>2D@HWGhdVE+eJPg1G~cF`7B=M>yaS`-Ha"
        "hqTw1Z6MGRWfLorK9bt(_S*lwGyIS!S#~zXU4aA6d03kt^F6<rVUi^IezPU|D3u`x#aQ!16l<AcqGsg41XI*-DP9pFiDCUoD%z31"
        "|3qq99~qH~wDg`4T}#DuktE6O&FxLAWKSAuXx&k*nI;_XnP~|R`86X|eH<_Cc`Iw$9a!JfqL!lNhj97L<B2KWGhu!>vHHS!qiAhL"
        "e(%_ca`ons_NR{J{^vgwYj}-1t<gXc3nTBD;J;YK1Fcr7o|+2GA{(YE<i#FMPC7}O>H#EcIcq9^lc_b@x~6R_yC#tQmQgb(_EwsA"
        "j5beil#&W9aeMQ((NZap`Gtw{M>VkQ_QtM}@2sD`8=bSey&>?i%q!mL9WeuFe8&X{SQN#3x37&nRf<s!H{geIt3e}s8?M47x6v@S"
        "ae{c@kMn)}*Ra+{Oy!QI?p_aHyL?W3O(=iFlfQD}h5y^r$4amdu_S?w)T;zYNz!djwwteW^3^7b=oo2eR|OR#ig4x2%{O1e(Tu!&"
        "eaX}i=x?lQL}H#$(LmPJvOf)mL23*dlEf1y5Vl}#c6vqhzzyAZ41<Y4#=sRL7&!FAJ869SAbH*rZ_IPO$zREK+5<vDn#hQ=t&WLw"
        "+RDi1OIoil8pF}ab`w2Cxn|@i+6~NxYMM;&Q0<52z#u&yYp@q<lFZ$Efau>39aT?YR~s0GXXCTGuqP@FIWwx>bCISmqR`37i%5$M"
        "EU-cCK=kY9UYC&oS2Y>)7H?LKtiY&M%S~lvt|lr$F%;9P<Jl9g$y+I0D8x~Jjq9Fj)^WkAnhInH@xb^4tL7;gST^i|L#&lV0Pv)a"
        "`-v*rGsBclv#f9@MgiT=!aSYeR!I^#NR#I}RIz^z95B5C)B|**9ayKSJ}`bbno%UM_5+uLwq_bV3?R$}7l#NA3id#|)ZdbWfjoiZ"
        "cOIA=Cc#1i0edjVs^dL3J{Qc)bsxAnx|&gN;mefep2nf-p$90sqXx_f`rSp2d7qO8EmOJ^ofL=MTdZ_{vNL*M3w=Kl1#fGR^q!@#"
        "HK(?hvRho&2JEOu8#zfr8F3T>hio;vFwJd*u@1b3-{uT1MdZRP3suhPXmGqO{w3+SHZgw84<d+F0GSX+;=s~5vMjw}C2aJWM*_D`"
        "c83kEb$3`K_=r33<B$jl0CG`dl*_4{ByOX^JgYi+;1KZn488p73Qe~GG<`|5vY?!~kATR{ce|Xt+`eBiRBOGBVy8aVbqgtFSn3%U"
        "w1ae7Gctb86I&qIB7rJOaj)8n7S@NL1k^*%v7qXJc>8jj6+os3=I7U3Vi)vuVxmbck(E=Roudz-6AJeQJw{wGV2Wxl2hkt~Pb2`="
        "zS_8qNPt$dPG=8<?$5VE?5zn5)&dj9$P-~(I>WszfEYA$;;dmvZA+3m(`32VDYvCLf}<>77VB*4vXpIHp%H@y!{DZG4m32v5Kp1M"
        "or;)k&Y>FS!~=}&_sFjfKwQ@6w)D?SO^K}!bUss;^Oiwxi$X*08i)a?@h58P<16qUf_P7#px;qY-znG=%aLIn{1_UD5^Rc$;)wl0"
        "kg3P<eLxNvQeNt!Q`E2+Pc9T34KH99n3Wfr^we9(;;LU8>$Qy?zj)TOV*fZ5d@GFKjRM23JVTN3<hml~%w6pGHGe$~`sRc-O@)M^"
        "ItK^ZIU%A4VlN}jH+_q~f1V!a2ojEeyywG@3sp4VoNn^YWr-9RBxDfQPNd;&OWL7pc?bQ_ffyJqxqy%z?4U*l9acke3*q#$NY2V&"
        "<l;~?7p2_d#T8M^k;25sAE98IBaq=*5A3;8V4LkDuCNz<SOL@Bf6GE3!Jl9W<_xi9h6;zE^gSCg4K+|9zIIPl4Wuf3Lg)Y^BCYTK"
        "ohaF{WJi)gGX<qKv2qIoOU_Kcg;CHF_sLU!OF{Z*1JfZ2Pj}U3L>?y?PGA+dRosIiXGEs17Bn8yFKFS@&yD>nsb?u=6I@Jl84fG%"
        "+ui0p`D`-G9?D;s+MhjX!JU)rcs<S{zVgflA;&HcoT7={QDP3pqe8M<ys(7P+C}ni@tE?liF};dv1*pK-b9ZS6ZFMfS8CP7ZsoaR"
        "8rj34Yt#K-w%>daru*fK&~855jYR_dw~%|KyLMh%d(>H``1hMFM)?l|<^^ON77a)^0S<m(o;A*qp^i+<-LqQ}C*NWoMOR{oS(iRD"
        "q}-5;H>#fN9s?u9mK{tM!30LU)RBZ>iW6=YF)JYu?f3{N!|U4o4P?=)oG;DOI7B)&25awBV|dQcAr{^1ed#5poc~;$TZ_T+(#BkJ"
        "SNuo$i}7J^^fjKq2!X$MFhXD}9Y1tMj(4>1+-{sTZ0bk&9M3BHo}bo{Z3Z^W%t^<J2SthEP|l%2v0aa#Jv5t^XSRYv4W|bowc&MT"
        "o3*I5J%3M(O5Pka*3sqa17f+|eDUQ)WGSBkDP_Qt-G8{Hb#E7ce(QOu<s4tTKdIyVgPmp+5O&6>Bq-H(^22I_8Mld_ljLXdPaub_"
        "ibixr@LV=)$@PBhj3D$n@MSh4{J+2XUkBGn%oUuCu+oS0${`EDYnN95IWhuv+!vvJ4zizy&v`^P`mdhbo8JLYO9KQH000080000X"
        "0CnPo8z}(*0OSGy03!eZ0CRF<FJftCZ)0I>XJu_|VJ~fOWMyn~FJE72ZfSI1UoLQYeUVLW!ypia_m+GI#5&3ix=6Cfs#2A9S5*kZ"
        "#HIoU4dbT1eX&zJ{ZY&YY2M>w-i(xT@6KqnKSBsvXfmB+Fx->&BB1R(<c%;r5k&TbV8{$sTpV{I50N`yXG%3@q`Z|<)>U1d34>^^"
        "+;-${3kiJQHsovz!i9vy;;0MiK$!E63xOTR{bX1nm~yF>bm~+hd@r3M&Ai?7%W+hmW?_W`p#f`MZ1enDr{o89L9%KQRYLE~u)+yI"
        "Em)=J05k0bXDA~x@HiQkfGen|*?la-X?o|ay4DoCIH<|C@}96SzV_BZ!UINzPd*g|Xd|ZEj>2BE>BdzlrM*{5Jc!r266MG{AIr_R"
        "es^n&<6ic~urlh!^KY3y@PC5cl;8Qr`deLn0Z>Z=1QY-O00;m803iT4l9bqU2LJ#*7XSbu0001Ua$_%IX=ZO@VQgn*ZEayMZEs{{"
        "Y;!MSa%p5|WiD`etyo)c+cp$_50L*rWG_-HE0?u<@ZbTxWi7gOU7BG7ib9|z%I2(FM9E3|u;0G#kdjQ%a@KA$g4QArzjJsl-#PNh"
        "$;mHOBg=wkOo@uuyb;NaH;pK0mDM5@wJ2i2(z0fJnQ>ZH0V~ReHEY2Vk@7Zc7_Z~CY((6&wFr-oj*qTsQ`T6SeYm`44X;<CVXYF1"
        "<pSRmw$qZ<WzJ+F8_6^IS*u{45&$S;b^7e;#p^fMuU=gLcJ(UE6JNu>>7{|UVhe!rV(p!UXHWfrsTe!ON*fs0rBb6CDVemm?(&|I"
        "cxX(OMb$RDO{rL0#A{x_(C>kR=eQJUDq{&8+9n5ZR*5>s0$&MMrmVtwQEx?Ze3Xc|OhnC=V$HV_zQZ5<sMl6CPY6bZ>`Fk{k0Q@G"
        "ENBFFw2>XY%&Sb~kQ52bJUKa`n`5!88<z8CeSD-n*W6ch+is(erzP8zh|c&aEDFY<rPx~zQ{22u2}oQL&hb&4ai!SR&mZ2Eb?!mx"
        "=Q3$C;m?nc7(TEKdoa4rcOgWu6LPb5-?Em;N_^k>Z7XV-N^G0EXZ#+1sYR9XSYJ5oE=$-*F&0IX9#Mp>$kG6bmCwmx0TY#)votGt"
        "gCsZ?v+sI~=%ZT1$MeJCv~TSJD8y+T!@th@t&poVA>yL%Q53mo0AxH=7D9P2!1rwfQ69IvPCEPU=AApQMWM>tXR{wh3zm&rUQ12|"
        "wqSQ*Ro;3RewYhhc(e0>ZG@<xYH?k+!m_;<D6@jyu}2+ht#0bZ8=xK!xAP%l-N=sRk5uX}s5+kY#qnInXfnE5nrZRbAkWMJltxjY"
        "&?yO@OH=eFe_nMN&kLO8t-{hvQy5|Iyl3@FS?=iSCejgpEt5nP6m58o^r6OdO&h4b#yk<O!k|l~C+yXe=fn{`Vpnw!BUW`;m6Wc1"
        "jk=f@Bg~@BqqX1(a_?Q6FCh<{9mnV`OPrpK{+EMdWTsvZ*_)KTLsA7jmnsC+_5(XT-6^zC6SBE&=4=;_Zi%k5<V%^!=H5EcF>JI#"
        "ETb*Y=B#5;M&!w8nWa2NN`(lM;dN9lu_W^>f@Oc{wX?G)ey~H=#bXgKho_Srqf!d+EGutme_%i=;u4%td=_=taRgcar%56Ey7ajw"
        "CP_+R=yBDeG(nEg#Zq|(Z3pOh#KJrF-C={NqMJ=b^fhO{uLbO{DGI!*viqL(7##eozQ<btqTs1NH8WV7v~d#-r83D|V}c0|Bs@{t"
        "fy^CmSqX>@qQ38fseJ33!AD4af|yRUi4^J?6mbn6N*GJ-X>h=Q%d*TMLY&o!X#OMBU&GCI1uSegxg1{Da@Lb0H@J*wz5njW^p1nb"
        "@KvVPcawN@6(%wz4wk8dE5VPwg^n?g#+sdjR8Z6!&cicys(XH@seW|E^_&$A9JQYM%%smZ?ES9H9DtjMO3<{wpJX9;?O7KMoI3?<"
        "DY;r)ayY~_WwVcDMd)jgw*MOv5`G`cJwCec2+MZiuWx7Xv|opcN%ul9cPP?0xC>4{P#ub=<;X8QL23D*O5FKPLj9t-_a4;b2s&Ow"
        "w=1H)-)P~N=E3V09aqr+!K!|in(xMa(6egeHF^e^MK>9u8y}c1{n&7(9o_kwi=Lv2vcmJuG5KKWl;PV9-~2I^MXNztS5j84a0Zq2"
        "cu$o9!Pabl%gra<Hz!BMMh|=5N2B}`vI?WRSeQ*srQcpmkCFe_E>r`pYCL96w2m(n%emTE&BB&;y2eAJEPCIiu^)lPgh{WA?(3In"
        "@u8g?&ICAhOQVW<x?1x}FmK75c&)26%0w|LSKpZ#Tvr^zf}Fv!Xyp*r*k8KiVPvQ20#gstWh)qo>?=9ff1N6)`sf1$#~z>EVi*y+"
        "ot~Z}8xtARMeyQ~$poR;2t6k4u&(smf$LH;z_(SR|L;fV-YH+_Rz2kO4@*lvvO2Ya4mI*lI<g!cS^at}y_3J`JErML-Wc8T;HDQe"
        "Eyo)JTY}qOaC<<|rkN{YJcC)Vch1ndsps67n@sR98GaLPrMUHM4~@HjFmC>S*kGVl^d{mVO)i}4Z3ei(h5r_goUr}xaJH+S=~?+-"
        "bVMt}wCmkCbzwzT?{8ei1G%UeL%B7~=pJPjfnit+%%8WJcY~K}>pt|dJp&i_<CvbiY&E!sUHtYW=v>E<4B<HsZEC?T=KJ;j4f`-f"
        "LYig2lru(lEk=(Y=Rx;wY5tvm`UhL7Kas7*SHx=*MvvWUJ%t1!k|>o~=KW<O2j<PeJDJHMJ@K7eOw23~>OsS{!R<I{Df43o9l8uv"
        "nKn7U^Q`0g_oKz^+&_41Z^H3x0Vn@{*Q<Jr!Q0OSO$^09K@%F+j8LQewDkpSdw`afJl@#cG-w(o*=be%SGS_WxI`tb+Ok!6HJc`{"
        "Dd_Hv9icb)qdD8_wM|Tf)(-^zk4R-PXP<%b2cm3^gO6DMTUFux5wO>mc0HNtpJxXz>Vv~{2IeZA&S6r0|2MU<6il$dAC2#MeDpU^"
        "O9KQH000080000X010WM3cmpW07C-+03ZMW0CRF<FJftCZ)0I>XJu_|VJ~fOWMyn~FJob2Xk{*Nd5u#`Z`&{oz6Z#EAbe_p#@T5H"
        "ABGJ>3k*Yd$bwxBMWSONvM7*DI`7wyj+><`I$*j879Zb(bX8T4@7_01n8ML(qzvBq9_0=~6ph&{bW8(ik3cCvOz9}fPU8tdFm<c1"
        "KL9Ora7tWPRkd6!7w3}T)cNE@P#_M4Sg%i0(({{@`TkGQE_UQy+T$>|TD<GJGwPiBfKHE$-$*y;!g$Y2JXJ0iA76AnUvQs21>{E@"
        "h{tBR06U(t;!kb6nD@wqW7Y;t%^bzKqBZ&cP{@KU<0*=n@**qX|77(s1kA0dQ<P@bSZ^`Qh44&rgM}nobo@NI-drF$*rT?@?ab+B"
        "GWzbH^lxGH;Z42Mv57n!C)Xh64%r&oAafcUIN)e4PslbNkt41pyCOa5TIql}{z6_qVchpB>rdaeP$)$|))V_n^9`<MhUfjjtkH0u"
        "X@ct%88AeZ8n*0WHs14kf%8nJpNcKc1axgFibcTXU@URfQ!(Brm!b`L+D>Nxnbfe{E@6P(XwXs4pzOTXe<{Cy7`)({3v@I60o5H;"
        "Uxt!!FcGg1jmqTzdZujaPEXlP>tm^_+vVanP)h>@6aWAK2mk;8Appoma~4$u005;6001rk0047xV=rQ9W^ZF*Y-eR{ZDB8MZ)9a`"
        "b1!3PWn*h_Z)t9HUvzR|ZgXa3axQRrjaJ`}<1`R{uhjoxFfR$$4e-EAM0tS&Apu(HwD;0oMQ$=_O%pq^-R><*|955_r%l=(dwz*B"
        "9?#EjK3fR!!CO^&@c7B!8Qr{qlQnwO&R7TDO6_X2P#X(=i_lplnbKH+wB=TLEdAcH+^jL1&*rm_vQd@v=s-3N?C`~X>$b9^|22{o"
        "1gT%3G~G)Ee;Lr|pN-u?Ny-kCCMU{4YpQ+2=7bPzQ?;G39xTphwKXkt(r+8JhA6+M-;tU(6j{_9Clj|>RG1DmADPc8tWRkcT!o_C"
        "V!7*#(mt8bfYw@;yS34{2#)jMZvo#d!0l95Q2dq4VQowUOZZhbjukD|joNH|!8M)k&Sxpy{R9=2J?|Z`w)#Z-f-=V@Yoy-n<pzuX"
        "SM*w5+6{Wr8sf>Er$QHOC`o%A{M1oLs5?t|?IpaZ3ppHQv&XZHJ4*6JFAt>;_A}~f>jrzgR1MmANY2?(4=X+lO9~m=HCBZ;J1TSb"
        "jGVNHr;q*TS^4d`?1}R>;X<YhsI6>~0k9OE7YnHHph{edvfGOkq@$M03T=K4MjoVU<hnu1(4kLyg^Y2Qq@$o)<($$R(ptspz6%k@"
        "v^n2Xtti2X4rfdICs$9CT13mEn0V1pIE0T}%lFnAo7CdZ7{VD(5TTgEYqYY;e|S2I^#9s2ggG?k!f_ZbN;4LMI#{@dikOU3N_#o0"
        "jyoGO^82pAB<!Z})iVBv0J2_s;qzv~*D4_&@gW3A`L@!R+mPTZg>=lpnY}F$UW1S0VgUH-wg02IwtkJ!>Cgp`64^vU$`vho8T?<@"
        "Hb@vjZ?#qYu9a8BasMEp)p@0*S4M*))}e_VNN(KoNG{p9&1$1bt>Tv0Fs-W*$av7Yeuhs?yg6n^&U`eMNbC}(;sQ{8p^J$76Rz_s"
        "Bybha%FHj;J09PNrWCHM#b9qBq%@k!Gv*v#fi^}qRt-_YU_F1j3h824zt~TYHlB;D-ZxG7y@63H8u^0OJ!Z*c^)7wP95!_!z1Jk!"
        "6?Cut{qLdw><a2xOMm<!&Dewmhd4ESF=D+3iiOZkmThvrP@;D-GGVWn`;B-*9@k>VI$u~}d-eTO3O@oDwl|;NKG1KjCg+~c1>lj{"
        "9U?@FpnQ?$jb==Nok_xmt`er8sV+nHOIW2A54x3M9QbmM44U^8BHrSF3lHHU3eST~n_^>Sbw#JVP8>WQ@VA+GSdrEyl-!3s<>cF^"
        "zlL9wB{%T%{li^LsfV}1C=h--(F_~K_b^R6!Jk`6)Cs@cKfwJT?_-g&+YUyyGPYGDeZVmudf@G?9={$$cop@bi#%SHi^+bt)e5qv"
        "up*W3hAVZ%#(|p{l0y~KTP6<i^~hE}t30(FLplq_s;EnPLfM`li^x@U;!K)j?J0N}{4p**L2pN_lBvohEKNMbl^Sx1s(g0eCJfHc"
        "aO&tya^mK}GCiUG15ir?1QY-O00;m803iVRC&n9a0{{Sn2><{u0001Ua$_%IX=ZO@VQgn*ZEayMZEs{{Y;!MTZ)9a(XL4b1XkT!0"
        "Z)#;@bZ>GlaCwzgZEw>s5dJ=C|KU_5l!!ERFeX$~Bp5J|U?A8xgd#V&wuZzGwgc<G=ls&d2?a#HB)0FKdwh4#Wj32V`*Z_$;Zjxz"
        "pN#5C(<y`xN`A%Ca4F!OthyRuBNe=rVmrT-!r-m(9+$Fd74A@IGR;p<PEUfT2Hmx7&9MR-Kv-k5ZxP<S(7;3mdM^ySg$i4<11^lD"
        "UBG6IRmH^?bjwO4z#D})*W4IXKv5NQq%!M=qC!#b0jn+Q%z<8E(=dyMTg4oEMwb!N0B_nF8>EwpP-nB5<+@R_v8-%BGLXu^`_{U#"
        "+8&23BUQN@uW~`i8icTBrzd62w6^!WQmoyD3`ms-k^AhW%hMB}hvNPtKmU~+5Bf}Pb-=fe=$gX96%K9**B$vOkQ=DDHe8fu8Zq&C"
        "h%)^{mTcYCOzqQrKyKJ=;V3QP9lPaC*Lb2-lG3>c!bI0`ja=QjsCcuCTZ`7dbDn@kn_4(~^zg#9+RF#efnE<DT`c_i{CQm02)a_T"
        "m7Q6_3;zMFl54i+H8=abPx~QR6>gv?xZtKJk^w|xy~+NZLy8P5OC~#-4S#%bo*lh#a9*50&W^C&)RNirMZO5fDa@b3wG<c?m6`=r"
        "lIA`2bj&>;GbX(8#>BekPKN#W#HPOi+4SFsG}`A5Trw(%RJ>&MMCE<2PH7$(70$x)6#2`8aWW>I4RO<P;MW`#BOSWP{vv}q5?#s`"
        "?+diuzJ(fpU_DKj^Eu9;&&Qv`zvTj%vW<4=<We?ks*Kg9m#$N><pCRHcBD>AV4ZzvkCyboqPQ8XIn%3GtkyWaLe2K5C|jxtFm`=S"
        "%MntmS7$iC`0f5dE6y_rCkdNrtQFGUPo}EBNF|bqpN%)9X{q8~x7noQ!)*N=+kGJpw%Oz6kIy3!kc{pBrZd#wI{d0a=k?XsZdxd8"
        "C5<wv_EZ_$lkyvZ<#81!Q*(M1AKXsO@vn<E^p4RYSu;~ok7)c})F|RPdF;+$2EfpmzHk2pI>3HJy04g$4&#*iI7-PNh9T_H2yu_+"
        "qf8HrLjzNxj%hd<(l(L_--ILp*X|cQ8EN*RA7|6r@ZNA0D(|)vznQ@zn+~lxoTX!+h-3eG?>DI@<9H*wZJ)>0AkgFhmj1<X5Mt7w"
        "MfS>-Z6v#q-7io}0|XQR000O8001EX>gOZ-$N~TWB?kZiB>(^bb8=%ZVrgb?V_|G(Wo>O?FKusRWo&aVWo=?*WMpY>XLBxad6iUA"
        "Z__Xkey`O3aO8y+wb{gD(GX}<0%=32>v%z^a_#G4aO`k)R<d8uIZ4x{Yd6uCRKD|f-}l}5k|eo4e>($7&iko!dEqSD09!{2LZ$89"
        "JNR}r2OmV`!DAsv1We%u5tNhu?Bn9{^OxoPa(TO$r@2a#BpD8dgAKVHln6o^;XQiLj7(vF6qDMs;7DevwIDoL8w10^hc0o%P!DJ?"
        "mQ><+IH=EHp080UX7|k*Evp6k(uCP?0Q?IY)e5b23TdUV{?1_SoN3}-+&@Z)f^rN<eg@ixHYF3pIo_2damM1QVWn^bq9{zYivFx{"
        "$6{5~G)B$s7J9~F(vIk)iFJHV$5tj^WhhjgbEPc`Y7^c}I)%pta(oQa_s`nH4O(i1PB*yqFO<Q<8aIv*q_eEIl!329$Pio+bgmze"
        "(r7`1_|(({?dt?#W#%Y_WyS}}akH%pkz_qNj$-c(=5efKhVpme*k}+&3*X=odY%Chw6iH(3S$^MFMWWGGsbOU{<lEV5#Xjk`I_gj"
        "%9%T9Id#@l(72y=`kFs_bLeQgMSYirl-pY%yt>%Md*`i9H7}t(!gpaxoD(@3B|GI;*AM|4mKhHe{J3{gtV<)Pn(T04t5wK;LXDdy"
        "<CDDBY(;c4`u3F{;IkYSxIyCfW1qzuA;L^IguPnFhGh@W32pFy%{gI^CLX3o>w}y7r{qHq{1pcFHu2mj$h1;u8>2ebEwNF**a1(4"
        "_gWq!+64M{c?b`$hw7-z`h0(R{G#HUdRIWV3w+a#lGEbj@chc4<r)Ro_&ufdK7xxwg0{@jjK)tpv>TK@aX`Zr&-btgmkx!ttBNnp"
        "o>Dn~W<X;3KGcgA-P*X5+h@uHAyLOVf;YY)9^k57xVuVcyZ6&Asm`$DFZzT!!cdYuz70o5-o>ObU;@dmj1IBzKTt~p1QY-O00;m8"
        "03iUh1b)gA0{{R52mk;n0001Ua$_%IX=ZO@VQgn*ZEayMZEs{{Y;!MZZfSIBVQgu7VRUJ4ZZ2?nrBzFB+b|5i2grZmd`f}T>0!HM"
        "=rCYNhoWndbw#g+BGHL9Su!Z4U55SmQGO>&x&rgbAgPb9k7SZ0w-;~cAU7y~lup|KCAlZG!b${T73ABnB?K#ienNOgz1{=}B&^So"
        "(j-a7qwy#wSAY^hNF%&Q54tEFg<cem8d-28H>tHCJXqVkUg_LPVb)j_B9+eOdflQdewIkOKpVc5D395DDe~MI)xNkw5u?6y;U0zB"
        "Er=XNPvg<YE~E*={6c&AKwCT;k182lTOG78`WL6`?9vE3ACG`P>|&UM57Y>)bP6e3Ikn5Y+mgge3%}vcS=={zcBK42txAI@Z4}lv"
        "1AVRV8$qZl4;eRZ<LCtk<i4Yf*$n_h2%H(qvLI-UZI1f|`>RH%J~^(p<)^+W6Z7F}k1&N(Tf>AX&DO^v8ZGsZV^iQ>`)z>=Ry)Xr"
        "WP0r2_TpVjtwHQ_=1Y280UoQ%Kz%yp;|P5`^CJ|f3!yM)PhOH<VW#(tCZ2gYKj7&ZoPT(MPj=W#xP%CiHqw-<h9=X@=X_5AXgr=|"
        "M2pf$JCvr!epeJ2NXr9npm`8os9>v+-wVZapW)2`aun96gxqa=B+p5$ujm<{T2bJ%m+j|2P#mPx79eQM+BdC9HBog*PQty?&wmFH"
        "i#GsCxS*LRBi+TyVA5+2JIclH`&mUC;U<a2=ggoW%Y)q$IEn8Eove%#t>W~CU2--(sCqxxkqnLYtS5BGV`L``DRlHFyRm)+vtQZ*"
        "^M2$_+}1N!hI4J(Gh&=j+ODVVviLf?nLSn%WMDoI1y_*uX}brOgdT|vKgX@l-1Ay(8Pz%}X7_Hnm@U67W;c)LaUt@<xLY_U8GH&4"
        "&B<!}3)c_b>^H^bvt@&WutCVR6$T3AWyrzs!P0?k;$Bml_|NR#^UOd%!IzVwpC=^G7-<c9&rT?&VSpQHc7sAIC<XB_1b$GDTeZC*"
        "l4-Dr(yo-vpb5>bK6cn9@q{N+Nbc$?4sLmSr(q-N8DAk<;81n0QKbnp;SW$t0|XQR000O8001EX#-x%R#|8iZixvO?D*ylhb8=%Z"
        "Vrgb?V_|G(Wo>O?FKusRWo&aVZDn$2Wn^DrbaZ8IbZKvHE^v9xSnY4)G#LLrY5&7R5h#H)+fIPcphy!8IPNyxZDU`isd5u{HBIc)"
        "cDm~c@wfBXP8vH+uXhq3s5GsKpV!~>>W)UEtJAY{;B`qQ%SZwE|B2>fqBZ532@aCZsiZuk32>o-z;CoH2^3^QB_37<$!LkR>G9F="
        "(I0|w4NA;)N!|hBIj9AZG>2Cw_fSdk7tM@N1^5zJSrwL*=4m7KM!dO9;LYpH<oJja&DIoN-H(k7#-C0yk}asRV)0A1tgJRJWJwD4"
        "o>+5BQdLNcmX(BYxH|pj<m&WW{QC|KyrmR!k*Sl5o1cGwd3X8q-Jds?X_>=ZNYD!k^c~5xF*L8Op`Z!Ep{>e-l}uYD3Pp`ZBjYgo"
        "AfyH@WVQ&OQqF-W;M|ZMALVoovzit8^e<!ebj=i-v4ZJMbbJK(L5M|8<#dg{<mZiyJIa-iiKEErAqqu{`Sh9bTs)tH!RBwc7M#Mw"
        "{zIY|#y<s&=Z+D*c||uaEDw#Iq&4aBhgLOYq*kOjaVluOC4r|x%t%pe!1`6t)Z$}Avw7%R%5zr2XA^h<2zuaCchs0=K9Vp+;hq#V"
        "y_8bOXmoFA5jc#sS}G`OrC~uZRUT$UXNv@3Sk)$DA5HdEXiL1u6np$|`e<~9mb51yG~}{z-egCMkTn4mdyk2lvpLp|uigs~Fl7tM"
        "2FO;5clP51PCuyn7KyzgZP+XnGYXYZ%oNX7bwT-)Ucj)IXOE7^ahd9zG&w>f;ZHPj+zB$TSw<(J4EM6p(`7Zv*vfmb*$3y3#;@+}"
        "d19S^JcN!1geU*YK?Oez$cM2n*{hT$SJBi??t^3bvWL`6h$4zxkw%r1lS!eJCn(IXU9ObAt1#Hn_$+}I*6~?`o}{{#9HP!&7lw@J"
        "DDo{%LcKADhLsbP)5Fw!4tf%B#I9Rq1yKra+SY$%hr&ghZV~6{6}Cv!-1U!JY{Er1EnrrNY>A19Y4_XP3)ev@ksOBsQJb=)8V3$6"
        "Se{evjfS*otghkBVLD};>FLx1#UPP%k1gZ~fo*!;QS14@5l+16f|6XhMmd4+dxy()cz))cn}e;;9`C^(Y+?3tTGbVmQJii)jk_}}"
        "ijwijb20IqN;)@^!02RaV%+Z~!~J9%Ta!sR8EhNa+fEwOw#wm)5VAws-quuZz_GSUa@JPSY=cAJody~WvZ)7!S8n*!qw`Sm+TJc*"
        "=fkA)XMm>lts!e4dz3BeNY#GrQTRP>vfGz-{Ol?N$)gZhEW?+%2^%gEad_m7I&01z2+W5T7#$S#nsKbp59Hd65jFXhAi1ZkAD@-A"
        "*r2qPIGnKX)fS>XqOH8;N66d8=Ho=J|9g?5*1JN7T|EqGgQMANoxGc?+iQG?Iw7=DlQTO0ZWpaj<oX=`#HIsCYj@zaDN^hnwrz6("
        "8f%vmQ<MrX2e`81M!dk$ylwMoKfNs(cFXkC@8T%?naF&96K_#bxM5Q@xZK)Y<)1{he^d5#?x)(fE}KuYYod@T4)O%1UYl{|x%DdO"
        "n7slH-3iuFD?X(5*5!a<d}cZNZEuHpP@B&@*iGYu-Tc<bbE`J)yu$8nrCo!`z*OlE5nbagFdEp)V`Eo%8d!e*;xRUEIg-%Gfi>6j"
        "s1rPhrxAS6>7h%AbK2Y^ai&)V!&S!$ZZ^Fg*PGli-TCl@{-wEKw3i#FG80^zdoIytyTTV{)zD%|d7#XY(RUgq`-V#oVviNKy5{Bf"
        "T449l5W1vQ(d?pUo`luz<nl&EPf%};rE5l?qkgT0lRF7#Z(joTFs^Wgp%L!fFNR0gl5SiiJa>DyBTKf&6E~!yj`u@eG`f=A-QquR"
        "WWC42_5a4gRhxuv!bPh%#Mhf+(X#K^I-M5+voFSx>T*%4mDJt)RIO?vDSjy_;r5!fq_i?Ac_(Y^DPMz~fN`n?a^-qc`J?wqin>y*"
        "72N<oIoZV`G<%s!T2Z2LiPuzOEv{+gr0o`jab^7p^cXesYYTGi`2{KHo=MpKlAXQNUm7wYy~4%@R@OzdYSBdSw!n6=HK2}P#3lya"
        "%pZ{HzNDr>(tJ8+MG^hexRQVqI~p@SAH~5!a<LXHhih`(ZVhm8{R_^+Y-u0nc9Ll(Y2lhL0#gyZMdq-w*9s|~Rl3*3=LIe<QDa6*"
        "_+~dpl7=T~Sl&~-Y)=ceq>(%D?ZvX15gP{PJKg*Qio)YP0G#egK|3>Z8|@${fuaP~!z{hpZ%_7Drmb7kyV+AMYSq6-d`$ms^I)JW"
        "cwd_zH-dFJYVAY1e*sWS0|XQR000O8001EX9+RSUu>t@9g9ZQqAOHXWb8=%ZVrgb?V_|G(Wo>O?FKusRWo&aVZEs{{Y%XwlomJaz"
        "(?Ae?kCcBH`6-c%sSp<_2p%e0iqsTD5j-Gdt-WJs#Cq4=UAHyj<C*m(b<(r~EXDC<E@#e}8AnmHK6<f&+8Wv9o{WY<<lfj8oH9Ov"
        "liONRP9C)IbdBIiw?I{`utM!c=RpqEN3Wjt7D?SEQ54N*^BJ2N>w%-?o87xaYY+~!4rYZl6=YPHT&NPON+h|_1(h&(Fa)#|(p}b="
        "KVa`I!*-*~+DKF%8(%j*hD*G9#=m3X1C0CLE*C{%lpLK;QTWE<h4G(J$o9lqV*}3lELXxgIPRU_2j?!)LhFPt8imW%d<J|-EFeuu"
        "lTXuyLsi7!82f6kmeD*MuHiR}WwLc!PTp~ZBm{srU7UbrOoKXS3~8VMZ~c&rG*_OIbgO~yFQ}cdF$nKDH;a~5!u{NRyM!licOFd8"
        "v7)C<U}fxsGtLX&kK=RxkKgg>$7|3A%D|%+wnPucf^4;@C?7yt$VI-vzCFgsR*=JkoG(IaMlnR+i1Yf*Hg`snpLE?S)EGO7mQz|J"
        "cC;u{l6Q+;D&xD`hlskf=mXgUd`*l745{OZz`a=T8!n8-1J?IL?%MvHAdT)=l~5tmE=RzfID8#{Wo1a`wjWs4YXBK4qf5u(d3Im^"
        "pCp4EL*8v`#qjC;1ca7wzJ?9^?nzS^dn2qIS~Uw}x!M`huI|YQA1ZtGXfuQWLW<fmuCs8n#|=N(V*AI+PaCD6oBeGxC0ah;kR@G)"
        "W#~Myjm4{mk&V^QXQB42Atlq=BJV5GDATH2AP1L2KzVQ13|!pFKdkG78wbK}n*?X683%XOhNMId(%fiegao`L$IEhqb~22Q)IQ;V"
        "ee^uA@L~J{bUI%%?4lhN3HytLqF?>jAww3NyCk6JrSGuH6n`Ry7N5`l08mQ<1QY-O00;m803iTvMhxiy4*&q9GXMY~0001Ua$_%I"
        "X=ZO@VQgn*ZEayMZEs{{Y;!MeVRUJBWiD`er5oFB+&1<-K>mT9eUSoLwcWH$U8vY?H(e~6v_TW}#V`bo#F<g8TSUqBwCH~Oo^wb_"
        "Bt?zgb^vEaB6)c3H!_>e=ErO(?>PGpuj`DxWJB2;S=kJH-@`!LT(LfDineC=yxboK5iy?adC#(8;0;cQSktnB*InCZ70a{yz~kr7"
        "o<I9=;B4%~(DST@@rQwlA<J)Bd*2B5?#=6n^<0$2m{kDt?VEQDwqZZF?_S63r=~ixg6C~vZuI<FDPT=kWjQZm_CBlm*{<#Hv%X;B"
        "koCM^Kl}oy=hawn!M5#iV1idW0-X1)fOkL(;3ojV&1QpGmWcPjeiaCR*8`pMO;v8=nh#l#4Viky?szXs&pe=`9s3;iR1<e)#f6$S"
        "9J{jF>ybPYgR$+<_W8knikpUIf;D=Z0Z<k|W(}KbC?u<wAAqOdwau>FN5-4Cz_iM<Hm-3%#Cw`4p8&B`K5NLq^Jfe`Gq`_EVs=OV"
        "E#>njJMvx)W?feu)4m<D{+M>HNO`?gLm)Sv4e9M2Jcq~uLU3E|+B~ZWhd3u%G$yYfI^LHxZ2rFEIcNwX{J#8+uPg;x^I_r;&HiZ|"
        "x^alud;akf{=UtI{1CAZ?JaKrPVqDEva)~ukkwt~g^IEV&{dUzWHe;C>O1w;w@fh_n7-#lI;pP)upP^)NPlg?#?re|l-sf@hocV="
        "kW@AxqqyDv%5${RiM|eLNlB?w0N=OW5t;h@S;2R#A&X8WKLY`@Ud`-_*o%ma{0c0jtysc7^dr3G4T*Dl`#snt53i*3z*xQ?tBMsU"
        "5VQfoz}^_B7DFQeqZ1us2voLUyI->%2&3(fEUWIbqhO-|)N#|8%nm`8J8e(F#?;eI$UaHf=JWXCtN47Q_9qR>O2OIBBQUe(ulv63"
        "1MRyEcg4U*++|e>vIBpR?SX@0unZwdB<y$-Dy<cQ4?r<lZLqOKKx2SYjPe4eAA;RRT1@&oGB&Q-d)^1(A2E6m4G|C+f#8JP3QWlg"
        "5fCJB6<Lv@GYp8|Alzg#4!g5gn=riI;D8d_=4N6O32WxVeOZ8GOW0?ZIEYBF1Q_=E^4kpqz2ofVt9i1-Bpy5DM=-yqF(7lBB7o%L"
        "@>R4pn1UUq6<T=;EEosR<U?5$ypf;?J!c%{1H3OvKtj%X$zCwuOaszl-<R%O<Rxfi;-a&!+HF>))A+>~#so`5nlYfCDMhY?)ILIm"
        "O}AxH_EM_>+GNMcCabCq@?M#Y<ob_o!xe1h$dZMhj?u;erF;nF{?>vR_h`}*-4Xgdsm~x<3b-5XM~AU*)Cy_xc?G(~q$#{5r!zV2"
        "z~jgbn|=q8i}zqa(qn*E=^?0stl&ir3XFjhLMlXTu$!-O(;T=3lR3+FIE}(9#cUN$5iq(S6_*W-!>NWCw4i}XE{yjOHA=)1LkBO="
        "K+Mre2x*;(TVp`Re+Vxgh!AqqTYgL>%M%ooG5o^#tpu7{$hiyjB=R!uph{Z&gD+m(0#hb}Lc^#*n;bmRku~i=dUa*%xAcU+;56G0"
        "Qtl?>&de%Sv|NG`V<+<rJ6Y3SOX1;V0%C)hF!1;m1Iz)7cs=kK?)ES#=3AG7o_`#{6b0iCT~+4gFqxjh4i%Mlg-FU!HAK9z0UI=K"
        "WgbxRW)J>{rC?$ftuZK5$1UCJz{j${S$K94kygjr(HMK=EhM#~MDz7%^gQq5$RSE563)l8fe;m_-(BDS8d!*h@m<O91HvZ^mq2{!"
        "S3rZ&gb=v4M!-7`4i<gmWf)U%gy}p0f-Y@kAY4j8h9ymmJhCHH(Ar3PR3ybQ#cP5g9XTY!{KhaPaGVY(43nUuwC0X@U>d?=Tg|##"
        "gLSumf^{<(FY}17zrs5FObn5<%n05+ir{I$GVPnxa%n+QQJ3$%rY+U|`r^j3fY4RZX%5Hog(=svc({q1v7Y@GMPIQ7Qd>;4YNRur"
        "r={#ANioyllPhyM^q7SHB5;YZtmYtK21<=IHLh^Oh)H$wtB(PPUPUJIPk^PZSO6fN?jVg%r3V8Ob&T&((?xZgjosJ;QX)u%3mEs8"
        "H-Qy2B0E$lH!9G7Lq&RDHbr}H0GIrnDoBX|EwNQ@-?w8YjD?zo^HRmZrz{qE+fx!R1yE;0JyzPC+G4~=a0bkWq*hV_^#f4ifhB*$"
        "7@~mDP;pej{}k^cwu4Gx2sUSBv)hF0nA{(D&jYn8&YC0bf6IA?qQ`0uv_P(j)RNygs7zXww>*#oCU{WwZAM}$O&mqL8|v(VfEOtJ"
        "M8YC0KSSr&N^Zn7dIF&Jyh8i=iS<nr@5V+Y;&B1RZaWT++q2e0tV<*_NgQ+5sMcb#avq-pn{<ht&_w;Ts21=8JQ*vj69XH+e=1{e"
        "O_7=s<$i2O5ro(yLRYcOA*>xJw<lCu>4D^?fT)aqZo2;g5<aTz1K<(-7%Q`6*ID7zdxIRFQp7EG*LE2|2BmZAdaU1}G=mazO19c="
        "0G)JG{1nQva!6C+F(_Qe^Q%SI#@A~w^Ik|vv3!mvV5ZuWDFGPNN%(R`bbl~$n_q#`|A3`>axAC=f+!{UPv~x^^U6h7eZn4Ut-Ht6"
        "T9KydczN{B#$tPLDlEMUsT%<%H*~Nz>S#ZOq+Y6JlQ?_mh@qLWUWver)Ivk^;1n9CQSqcRpxgf~(1Ib3gsMc*+a`()#44EhrfhiD"
        "2h)s5wHes9%!I7&9SJy(*wcal#2>&&G4R0y>^DMHg^dD>*EBA|t0l&AWkT)VV5Ma$Cxy$!bKR4yRP`Q-M-u5%;HVKLBfIQD5CSm*"
        "qs~KY3}FNrdiW)kc7?)uEhf-&pWc#kRZ}h~{kRg3l8H}q=f=~D%xF;xt~b4J`$7bv<&@_4IL~r51~Rcc%y5l|DztRK2C-06o@5nN"
        "e@^ZyMjW(m>qLiea?>r-Vmy042|a^$Dq)m6-Zgl>l8?RM$*R+p3(TqN`!u`tkP)v;JX|$X@nL#lsaAv)U%77*tKTf4t9Yx?UBF#J"
        "lwT7}`SquWk-KVg{)u~(dCiAITNs?yHN6Nho3XJbF!UZUZuWq9KGtHaA(0B%8M}bzDyzG+E}P)oQeLSF7*nW2Ecr&gPbjb;sj<6X"
        "PW#B!P2H4P^R!QF&uo;>I-|T`n+ZHM;_QG&RIQF|ZVPF`d8#eZQJ=ee8R^am6^|Q!55;%~&VT8K+?~#EAeHnxmep8f;4bGl!OYoq"
        "+A^6%<Mc2rIVG!2OZ-gTAp|JZb5<@#;LcLFX>BA45Sqkbm96vOKS1uR%cv8j`)3>*^z4UEe~HxjDUMO93WJmk-XGeklD#!3^OTCl"
        "{@it`ZK~H1I?2C9UW_SX5MIY|EYsZ^SryFoAqARN?Y<nudTZGy7VU-|3201DcX~;RC)&ouY|_~jt;7<CX{ECJ3)=PjAAcp;GJ3Oi"
        "Q`VD!I1?WaBqy!R;_Bb9G-Aj)vdCvZ)sz*XMb&E%crozg(-t^S-<sgc?%5}a!zl&&>hi|J)TMKLt%OEf)uR~T7Z{Dxp0?bhr3e(q"
        "p4g{LY)ntI_vGP&Nv>zIeNwSvACADl_xp5kN`o9G|EVJkf1Js)1|dLN!m*S+!B^XLD|-H}>psDjlaAY5_jI?7ALopS1aQimgFqbj"
        "<8XT7i~MnYg-wSmqX{qFsb-u?khQ}$PBgM^rPkb)cs@#CQfwN*%+6|}0FJnNT^lA8WQS?UnnR%8eTqx5u?Am@Xd~x%LPUcH=_*bv"
        "(kM%)-Vm_#{GJrgV8t>dS=x4bOMACy5lt{lw(9?xT7AkYNu+pvW!-jxg!3EoF3jU8<1*$D*eRxaQoB!<sU5LtVdCFvtDesDGN`RF"
        "=FEl-Bk-FqT&C2=h^ZJM#bmcFJf}&%d~V9GR|tz7ik&3P-aw&qYQ>tRol+eMPixHrH9g>DmKKv%dYGiThDInmi+neVrZKr|m2H8="
        "VK`qjlD&S_3d3GkN_U?EUoVhU(dk+$X(_DSmL@_mHtWYmZzbv$JK5fjW5MsroL^x@h<7LQZrsq;7;4pix}o2U;I{<lMR4(@zKJmw"
        "+2n^Ls6c*&F?PA>strz>Z!zJMp?2!H23}&%mk(*jv)i=hb=x1w;)Mn%*AU01a}z%=sn>3;QaX&vbRAqq>;nHjgTI$y#OTH6;f>o|"
        "lufd0Z0-c1iU1^9fMQ>33qnI7uWjA*-=7OM;@Aua4%PD6#U`2%q%4gPI{!RkU*gsmui)t`eDg)<2FJ&NUz)%#OyIAG^z$WPMe34^"
        "EtvpNFIi2zFfN7lgPH+_yb#NCXk#$mX8A4oVzqv{N`oS*N-+eqM?Aj|bn3eAv!Vq3C1G|kK2jI5!`7n#2k(f;Q3Df2+Ya03c+{0a"
        "VMT?M)WcH9kzgOjI248idXJ!|eH9)}+hC?m9hX`H*~i%yPcQIA$i9Jedwn^ehe$d5PCP<SqHw@G5vMAo)sx!(X*d4J1K^=gG3HQ@"
        "mW{k~qC703sIi+OiSjZ|%FQSDk&vn2#V12`P-w}<v`Tp!dSlg=s8Wgi@|2I%>`bKi*ZbqhKWon1{rA+dCd~Ef%Y%&UStk{50vjXM"
        "%w9p`+=Igcl+zbCD`b+3JS|Vi8Z5n#xrDi3R{I2XufxMyj;iA^40GM&CsMS^#I#UQHBe$>?xZ5ARK!Gnd9g^b36d%mP4&t|z3?qb"
        "w@1t;ZR}rw1#@}3j|B8ML3%3#+};4U_knZF)5H#A5yqwz4_@&yU!edSVXMv_biA5oX}_rho>w_%=;~hKY2bR&lVD@N*0@2XZX+d5"
        "-eha;>ct^!*W23f>%PNsTdz+Z4}?Aok;RloRzh=*AfB>|$zyP1cuEaZ)PKlGu`dqL-D+|Rv>ocKzfJiA&&R=;vX;iJXK>SLu4>#X"
        "zeg#S|BAqT6#X4k#X~>l1Cw`Zp~lCXiCh3Yy65^eQf6I71>5TIg1EpO7jWF|@P9E(_s3!SUD=Lv)wqXTOPy+`gp3ZwLw{W4XJRqg"
        "y~IFWPhTGB?_&FsSM#8G3e#XxPeDhM2MhlXP)h>@6aWAK2mk;8Apomc;kJJV002!F001Qb0047xV=rQ9W^ZF*Y-eR{ZDB8MZ)9a`"
        "b1!gmWpr|3X>Mg?E^v9hSX+<WG!%ZX)c;`2OH!t3yNFU1g0z&bRH>8-K_ygGk&`$xW|KJB&U6RFf9D)u5@#kmTO`D;v^x{~+|PF|"
        "p_IP78Cp>ZLwaV!o|9_FtLs6E-jG(ZTJ&4O`kJts4Th_SvTqMW^bJ?MuQ*X$8>xU&_8F!0{=NJ6L^nuf$d2h<E6VBn?^^cv?=?zx"
        "WMBr5h(CD?Un4@`SMB^w%^GfaucgwNeI95~wW1L%M?!nG6pgG{yX9TSvQE~#)!A%IVHE2%XdM`Y?}gzvrkF3D;pQFW8I+@F>%u)N"
        "?2R>25F>i<Vb<|ZszWg-SxzjRWkq=~T%RQ#B!uV?$ST=4Vmm5cxaUDS{?@#4zG;!oLcQevJpw;^;MK-`ta$|oDze@mKxmH+@;9>Q"
        "{`A2_z84kGX*Jf2hG*t5@Z5$fD;v<nJbCJmjOY*f6>D|iv*WB*Qg+2wl=-vlv45;Nuk$Bg!?*Ogdjg4dHMT8rae$orV-$*$7r*{+"
        "`DjqQX~lMDz9GFNwfxwl`I<;YUcCL4K$L56Ys1>MWYx9H(RJHNZZk1CJxOo#N+3BsNr`CI2!sYT4MA@bu!Y-Cjy(|I(W&@MXq)Ou"
        "idOkBhEQf+hz042A}G;H15#vvdShnq?F??mAILyZ@(PaMNOL8}zJ95cR0;iFnq7!QO>DLigc8wmR8?F?H0eeSf^UR28BG_)o>*|+"
        "WhaWgW9`U;MojZofTdO(k|<trL@etA-$<=kl{gd0x8(6zIdAyZ6@ox@Y!oJX!~vc<JUF~0?q8ZAQpjgHarg=GaaZvX#)a*fXjzHD"
        "i8!w7Wg)Usy?v0h#fwGBM~;Be1V90CGrXQG@cPz4w<VC*bu61FPYXQ`SetnLsfwa@7>`jwBQelZCN(Ek<ohd-iCRzkmB3|C4tb{q"
        "II*?O6&n1rnMzd04U{Nfo$O!F@U}LPOp-DW$jw^0Bvbh*urcPpsFBaf5NCJf6{a;;Ye<Y!eZGFHic3Gw3VX(z<@u?&qYL_448=#O"
        "YMoofjo?Kf3Z~oRBf1qAL^r%QT*|pt2eFGgU*zh6DF&h9N>9ZI!aAFF4rYzwd_xM7$LR@Ox7c!1z`!B{@D`blRDk$}dS4Lq?xH3<"
        "AQulFeEIm!++H5z4}I0^v1wq!f;4yH-u-;}BWVyzu8ssa4(+I?qGEyS0cwY<CF_vB*3^#mBQ$$#oC*H%hZYn64k+&58c;em3|LKa"
        "@)z~lK%d{9n^w#@=%j!D$1w2<zNemme!x`tH0(HS|3}=!5y}drus0I_TyZn1-W_kqUo#!$Hv?CqgIogdTX8)gW^;6-g6dhgMy$g<"
        "Ba2#8CQ0MV-f3{he~J%%UKX}g*Kp?u+SExXL=EZ)b13$pS|jY$c+OkB3jySCkG{5(1I`~B6y33bKf2}zvS+QRZRW#F$6dT(b_TSJ"
        "gEoq@4xF?X8ZQK%8I1*EUSo!0dPk4nzJ6JN7`3kDsCiK_2=&q7-$&E6ZcO1fA{d^al1&>?;)52g9<@*@dH5Yc_s<jUJ7=Mvl=Rqb"
        "U`p*odw7K5a>Ecf^x$hHDLKf&+pOOr5FNVbiaC$AP(G%~^hk|>Wpxd`H@7=kX6NN3u~d%;{rTnP55Hx2LemT!U0dw9$r9w|bTrMw"
        "XEfAjSI0WoX?l{-;FBy`+@SA5p7o-ERnhZ}P>#DGXyT<Kr}?ZFTcCkcze{EixQnlzCP?E{a}6X-vmL*2x07Xkccy000}|w&;!xu*"
        "-Z`dRI$PqADujx3LU+tS>7|oPoq@MF^~nBD@T=9`6QTAL91S9=wSdNJqJ>f^<+aVq{RKTcy}p`OcRztmb5~^N{$et$k{!5Pl%fTM"
        "fptNt_jGYWKTK*bo;%&6s%7<eb#E$PXB7@%n0d?<!N$OI3kv&PIbYKwEYyB+j^_5TVe7m1j2E+?ATO~^^Y98K@b~N{ojxZ&;cF9%"
        "4#IE>&^uDXz5-nyK~6N(KR*Qn10$d|{OR@<+?wn=80Us;(wOy!ghHMvR$Nj<pG9GSBThoXcM$92x>yWXXDxgUnGwC?37{rWYh&e)"
        "93RG*e118TK&hTEuv#0^c}A~>abhK)C@hvHH^Ikh-?0j;HTY=iS<EMyd!v%=0aIn6(e-&wo?JXTS|CpPw*$!9+n9N%-S}h(FA;ou"
        "#zD&FJ~86}o*FXS{L(0b14eKjj!&a#6E_P0F1Cu*(NyKuu^`XkZc$New&HSGAZv&P?spnad?YOiexG=J5>O`%tr_~XR{G?1c6(Xx"
        "@$QG<1C)!OB4@boqMsa?3B$WPQ~6XW$7oF{tSB&XCs0e<#ATn*4dRg}E??6O&r=s9-<LrpZ$q|6!FQ_k1``IE24GF}^!A<SmMrtb"
        "#v;S!3?aFlIg(qezzN!VX|rgt#LqD}wb;ktsF%(%d{`?v918D47(|nk3w|}bV=VS#I=K00P>v^L-zT3(oytG@5KnsBR+hLk89prT"
        "#?Kc7$GzjZ_xd~`{sT};0|XQR000O8001EX6-Rh_n*{&>BoF`qBme*ab8=%ZVrgb?V_|G(Wo>O?FKusRWo&aVa%E;^a%FB~WiD`e"
        "l~-GD+cp$_50L*r=t<fXHBPcNGvER1;0^_vc1VJ~6oWuZw9Q2#HIho=729v$Iix62ikfr=3`jf|p3BF>Pf-+Iot`g=;o?9P6{W0C"
        "9vI*44WU}IdQ%<9Rw`nc29+IYI4hV=iIAW`er0t{$%Ynp8!1>8MbYum@e!{ZsSGhv75nj9CIq-8LZI4K$(n5VRu;6{p`)yjV#{}="
        "v-s`$)4TT{);B3xKQ>J9nhA5$u;MG0M$y7(Z+s<7R()zs)0&jru)ogW|CX9!pORa7#{~bImDjAHT&*8z-BfJOYe@~dt||)Ad(Y15"
        "mu_{=tj4`{2E}7#2}SFJqD@mB@&~EPycgz;tax#F#!t=Ej+Hqz1`5EX7y{Wc#YDlbtisP;KIaOG!v*Y?$<3dvF#MkNK{Z&Fyiii>"
        "92_Xw9I!jbM@0n)Ah!UFVJD0z79xX(wqnWh_=v!-WLuKwuz|_*ShH%I5)F-7P8yezdx&V0YbgGbaAC-bEY2>y7yRiK4B{l~y(NQ;"
        "Z>`r10)a4~ADVmw)CJb~gG4nWztgH^Yo(-$Bg|j`GwgtAYhv~ckliZM`8z3@^LtMfr^4tg3O2{AGTa_w))s_b^1w^8*RcF0TA#g*"
        "mgM{_CDHl<A1^Nd85FyH@{-q}?aV%h8D&_QKL&#?pc#WsGBvP4`fU(q7#4JE?ZFlg4M@XyQb2)2BQGCUr_;QBATPQMB7VVTRp)y~"
        "OXutZ7mTWSWF5NQaH?1DXr<ZI&GZ<7zh@3vJci2Ca-}m0qvYolx>P12!&%-}*tuBQ-pvS}_)eaprlJSHrY+Yk%xMin83x1JNk5=)"
        "lHI7i7tkAC=Uc`~Fnpk@TwRdw#yTY&ec77TX?6-Fw^L!ul{Twe)w0>x!UBFRw0M+2N)`w_x~W($;7WnH22)E*HQ(MRqi^-qsrO9w"
        "V@Zgzu|%}4lX`VIBe}cpg#KQdIf|KAn%LtSZa8v0O;+05Di=^aSIq0Hn~(4ko{Izrv&&*jx=wZz*LG|<y6X~h@>6&dE~l`<^A|Sp"
        "#RC%CUHUaHFk)WnWO^`Ek>9aH?q@(v_1)^<v2ov3E1fTlPhA@d48%t}IyPNMer*V8@M3ogvYo;zo=1q=G@S$1^t69tW`nBV<6DYQ"
        "$sIR+^4++Jr<jQ+wfeqKI|+O#W?CgeP{huCI!6S0nG3q28z3Y};`jkVP_-gg{zT@w25J@0U;QP!=LKB*+Tf^&il&V$JWF19TmpUk"
        "BpA^L0e7?Eg$MX6B$)V+!X;?-z&%L=c_5{Mo^E-CgcP|N0hXo<shD<D6rDQWSyR!{D%UXo;>DY;9+qE0Ff$Yfb4W5HV;6R3z#LV?"
        "R(u{xI?pHLsL$hM5v3#sR<OYTIs7k@ZsGKYl)S;0#U;G`h%2v?@G!mX{LDKCSBux7<4gN;I(FL81=~<NPD3^uC$Z1*Y4#3lU}Nj%"
        "j$J20b_G<@X=D{32-z8aBOpTy&8;x=4fXV{p^DbbFomV9xHhrfBOBIZ58|DoB?na5CiL<Qm+gm2zXSv`^b729q8&uo6$Nx~Wg}%3"
        "JKS)+MIy@LAo?UhoY@bcstQKOoSBa{$xs(8F+{(J5%%dgKOq-(O!m5k@lbm%+ryL;vW9@Dy%ie;bi9DGAn*eUHzpY9tKNA}WxM15"
        "W!NQR{Xqr7YOgc_RSF;&VJ|>7U~V|_QK;w80-Ek26uCK)>pg8y!M3<BDT*>H3Xvadhsl6dh#<(Zjz{Wa#%K3ZtOPR$rO`E(b)UEZ"
        "0Vf^ZD4leiT1R5s@SQsEmjAa~JHsp)8jy-zXzO?YE07O3Zpm}#9u;Dkjk()VLjh@N!?i3cRNg>c_BB;^IeTP9Yo2Jq2h4v^O9KQH"
        "000080000X0D~B=k39we0J{(X03rYY0CRF<FJftCZ)0I>XJu_|VJ~fOWMyn~FLG~nbZKs9E^v93SKDseMi71HR}9pLN-Ztl5+hLJ"
        "B5|9vMJ^5O)GvWSFUgU#5XohCm$sFn|K6D;mp92tYrwX|ot^8MGsDSb^0O)|K3FJ)w$e&f$YKSeNa+`3mQ}`TVdWh{iit{5gTZWh"
        "@b((eSeXl}G%V@&LM~MzvR^UJ#XJZ;t<fM<R%WCV4WtSRWg%076oZq3SR|WR6$o@)=4FOC7M4ymmdZ#{yoE$0Yn%sHY*JWD=j5Qb"
        "Olt8V$*L3qi^ez0k3&ebGN!u`x`1*L$>1L4YHiItm`o-?u+%DtR9KN@!Why>jwo%rN413AmYkO6@E4ib5dJ7xwa7yFRFU&R(43Ov"
        "$vOyvmz@sx7cNz0F}<YdE`&|iivZ|{yr3tVvQ=0!2>u|$JMuSXSs)225T-8I!ff*#t(J)o{J<9nA26xduel44Ei@vLEXvA8GBu0Z"
        "#rY>JjM4*{He*pP%w~^ZW9O^Nyv#S1wN{OdD8op~d{RI6g>T>-J`hlLE~Qx7^*3(?KL3>32JNmCx~;%Rk9`BCD$7h#qOICs0oJK9"
        "F72gOR2w?_4@rN?NR$axX|2*GD@%fNmQ)!*H1av(W{Ng<kr8(UFf(<c{6og_(nc8;3y{<n=Wu*>`s_*A3aqf`pFTZ4>CTEOk20}E"
        "ZT#`+=#)&ZQKaqpv+kImN2$#H(aF=Z?l9g`e?=vEPH+^ROf4~kgP&Y?d?wZiVM*<7Zx-$43A0VkI3AKo*V#lXs>rs~uf|k4w_H^a"
        "vFIF@RhIPx>*(e|tNV4$1g52ewM<hg;3JFxZx63S&jn;IA&L_Y0bQ()1hb52jY&@3q2|=gYxeryz<L{NNg}OIzCaO}wa|`4DNZ82"
        "R-fY^Bn3o{3<ptV-0~#V=h@H^rRT^*6-&9QwDYYO?kl}AU1G<8@tkh(uZr@4Ku=e84O5)2<^-e-Q|{@>rcbpF?`Ae1C~ZOZu2!Iy"
        "{0PY%Y6H$Qq;k20Lz*7UQnh}9;(l#?SbnJTn5skyDel5>jp?h0g)i^;b_d(;{sy-PcOt9MA$N4C1M_b+9_TYU#?>krBwWzJg&lxj"
        "@B=5zGp`v^)02AWy*WNReDm@6Y}PByCE7|Cqv~8WL(|AFJY))ZFE3z9t{bRmP+CHSsg_Im0JpbM9^c;1>Y3#Z>bnDdGzzN3Hen>>"
        "l2{=FOOkTq5M-4V8C+k!dJjY{4tDnnkT9|xXz~~_q>~+oFQ`@HCwt(&w&v#jW^O;rWe<Mv0*)A84R-kJFc(@#3i+B#_F8MDr<1E8"
        "P7UPL4iIzu6a}taU(6@7zC9v`X@{q9fR3;&dpNG=3|^dn)1Seqi}f8_A;4D!4Sp(P&rFA?x?&z8oWV8@*JUn<f$mwbhge<fxKA!N"
        "W>@%O5FdT@0n*1&gq_Oo;LrW!84H4xEO60O&H3H+b=|uR(gt~5(f!zBl8&5Va&OwJ{g1xvv&QGTJ_9#L0q$3x$E10WxkH#9ADuiQ"
        "h5T>UkM^UV&xR(rUF=50YB;RXCi|RO*6}5^4rT`L*Ql`zI<8mS6%#TK?ly3b@c$j@jz$;lBX%B**tPYZMA4IjeJ0wAZ8t|{!qa=&"
        "YC0TinmMya^CRxLu@%l0&{@#a>?q{#0m>(Xdw%1|<H=&IF&y3-$1vEK-2FXKj&$@EEEghO@*f|O>}98=R=PDvT6guIukL#qdD)9)"
        "Dm5ncjvDdIbsKTVGsMB|E*JLt;Z7##_vZRrZ_n!Qx`Q3-z0ez8T)JI$XUqc)OqtRGBx$JAzEUEAEcAZ#r<orqf<}Cr(pq!qu1dQF"
        "j=^s$rI7lbX|!V*KAywxwBi&Z$6iz>(K04r^K7Glu}Hg}ny2A&kn{qku%2V7G$PKBcIV2_Kx3`sMXQbmjbVRi-@2DYs_vP%jHJzE"
        "{F=FcbTx>rlSxu$+D}9XDZiMU*T=C^^tM+Q;=KMh`&#+iSL&!Cn23lrvO=%WcEx9M-hGqmVBpw{1Ri?uZ~$<xor71NaG+wid0(*_"
        "<E(XiAt_XC>OBkcH1HJJa46Ta_Iut>Ui)g9G9CIrJ}*L8v|iur@cXWG7vjyhFg+7tUqkROP)h>@6aWAK2mk;8Apph#QCr>=002l;"
        "001Na0047xV=rQ9W^ZF*Y-eR{ZDB8RWpHnEX>@OLd0%sMb!2%iaCy}{Yjfi^lHXI6|G?uXGLhuRWRi*Zv~`=rvp02@nW^z)>#7_V"
        "3nD>@V~SJ=(zY_?|9;&KfCNF1lAY{rol+BvB)ZXP^czk2zW**w^Q_=r#doF1>ayB<qHdDCw}U&ZVs%y)GjGicF28%MNW22R-f{28"
        "_rG`|E~~6qdq4j7n}zTDUw-lB7uj}KR<*~fH6ReYy|H0plV_{;`jaS&_M#N+SFwlhX;p5$xXg1N<Fvw44}UD1qUMzvuwyl%)DPdo"
        "B~n1oR#{rcEMN2OmM!A4NVBybclkf>FF(Bb`SR_@=<VNrzP!Q())9Z$@haQ$q8`CeS1hhuN<IL8vg-1IZFf1J;iY2W>#To$Rj{4d"
        "luoP(t0ijwP;107E6O5+A(2p!5wdO++}JvWA^7qO4?b7#e);g@W%S|lA8)SSynA~!lehn#)taO$fa2=5+;ARs=-?~tZla!7oF$Pc"
        "n<|De<-KTcx5H~DZX+ZTpTpyDl;pa+<wf?plqf}XUIP9**nu?TNu);LPUC+GPra5KGzCgefct+z+a7xJzdCq+A&JCEy(LN90_PC~"
        "xu7K>>jPfIWdd@%^qV@Jzw*OSudZ&Y!hu+^SR=pokq`dOOkJ&*;89#|w^@xh>jq>b=!;13)mJI}3B9zeJpATmg@VQ_;om44!j1*^"
        "RaSF*>2l6cEtM9A8(Gwg?QN1(fxHmQk5vP5$saONN98SDh6aqpP23<pCCzpx0z?f!g=j!Dn257%`I_Yd5M@OIXqV@q_qFH$TKG81"
        "Vxo)iyhpi|F!5&+-))TVFFuu7VJ-&+)3F-82HFc4gIl3V8?I}xElMkrY|TY&3Z+`seO7NIo#l=ffnTls&|?BDVv*$hqN7-eo3fM("
        "lmm@bj-7eF*s$}L-}>QV!ynW$gOrN@)?~1_g5O7FQ|}rVJy|NLA$E!&B%{uXz0?ZeWQD8`Lz6O=3GSiXUJ}q{Rh3oXzoRKU7!GzC"
        "EfC95srDY)G?szgW?t14fAQi?Spov)0b!mbY+(vW%IkR3X+3O?;6R$#@puLnG2U=3GGBo@;V`vxibn7~?{@VZ_HD~+hEjXdB9q8H"
        "kiGp3<_8<&ZOeq`-aXIO8=$_tFVHGT2Zq?lY;BtL8g@Npu(g|}buyb2uObbE3qX)9E3y<eqi<$91_nrd5Jj}7QK+QcJMnzE)Q;Rp"
        "qgo&$*ysr})rulfjl#=Q4v1s2CfCTmMFN7H@ahI+Jdk$?4c-AkFmHGqIO<*Bm>Z8?F|K(6>(X>xx05W0k5QJ~80N5*g{9m$Tw}%V"
        "efT>BMDLXYhXelvEQlXPdQCl4M;imphJtNjgj5gYgkD8!TT#llAo|8i@gp@ED5u@(Kf{l;`N42OGc|X`?=k`IZP^Xcx3mD$|HJ+y"
        ")}?Y>k}N3*t9{KyV2y)~)J8+v2Qu6}8{z%iQZeLWxGt&Q|6qB;m1ciL;yjW~LhWrcv1On~3*VYM=yv>s79T2eEGZ>IIz?~8jHU=6"
        "L~Knz$|0<@ABY8Zq;`F(y&|j_!55a{5XRyrZ3+Q*&qz`gY8;1gsHqv5@QSkb-j)TIgIds;7XjTP(Mw<=nVeKbs6FdJ!zmn(<l`eE"
        "7HL}n4*@GnTX9k5cRXmLnt_v$f7Ek<VIiQfBSlHjI;$n+u%0|G?*UDHUsktEdfL0C_o@KgV&-kK1Y!o%jZYjS=_S~jnfj-kIdBCd"
        "DR`tJeND6AAda93ybcU=he5~#AY|mSdC_b!PvL<btV+u9ThFV35D^CNuZmxQD9SU4+B`s>gRhpPhwT7*XerPNX1ncf)z#h_fPpd4"
        "J{BsgeI{yYEkotT@r#@l>xQj4YO=rE*BkH-N9i);6OkNG6Wa=m;eg`%HBE%yZrUa5XOU~|?g?8Ef!PsAsf2Y*O!GmUP&2TQU_Nmh"
        "ajTD-CV{XT5+yJ-1_s9@lD|rlg#$G@1+-kSNe-irAIrK1EuS!Q{0|@gPj`&CKU8WDaa2E4bD;+$(%TOoJ0SE@P#=a#QGc~yllAH>"
        "Oz7+#yfBfY(zmE+%@z)dr*~C$2b%x@qdBC5tB7qjKE2~nXL5(nR(ToUg5{-|muMxV#e`JMd;&?81DNF*?3-9>5E5N=6VkUKz?MBk"
        "Dr=XvUP$Ao?~$}yaJFZcvB`2Q-4FDY1=kM3SRKHXy@wd9-Exey0)@X7EDMd1NsjLU&EZ)7XSWc`9QdLIC;f1M(^xY@Hb9KqV~{d}"
        "yYQ|*`$!*rOBY0_=M(>nnr^A@k21d|yP_u{46c2BoZd5#_G`b>IKblHz!r3zKS4uabKWvX<WLrD2bo}k`KZGfD}6Fz4u_tqQ7(d9"
        "bUI>L^2VAHh0c~Xjw6yJu<f)p&gDP8UV3L{9uf0h?FF6vo_X)C)R&JC?SUZE6=lX8zgde`QG+J<31SHW3AuTY3mGts3^cvS1v=sd"
        "grh<4_RP2quEu3e<eZ%0-h*V<RY0o=uP<Jno;sz@DRfntqe-=kc-Ka-aC3dG_SLMfv=$-oYw(!`Nn!4zWJ#PLvGl-~`JT`3!1yS+"
        "lw-r0lG85OmeOKuiE~PcOUaT?RtW-NjaEm1Z5T=k_0rrpU#<)oH@R?wi#{$2w+sH5w^@t=$8jf(K+YVotIC)|tZxkg&BZGvZaoO;"
        "LoPtY<2ukX-x~qSR%2w3!D~BGhqBeRmW!K(fY1em=mMB>7D8Ghvv*l|k`r4qe^QljT&f&lKJR%R7&7UO{NEZ@v7*ijt`dt#Rh2I6"
        "KxZTcQAubovgwfvakL7*Tg?3`Sr?F-a3DQdLS<n9@ImeBEJiN`c^s&HzK^ho<J${o+pHj<2q79|k@VrlIt#RfO`bz)!m6l?99fe9"
        "!%U2=;S_PXL&@@g`S|)3WnDCVB#%EJP33Vzvu+v2IWngL_T&*HNAWEvMZ^l=6%Yr3kjy^yOpmtS8aiYgy+l$+q$8OrjnReADl2-m"
        "TKUvj-*hD4n`X-jRM#|n7-Zukb^B7D7DXa$Pb)aE4+DG|=S@Nn4=TEf_($*?x|H_RQoz3?Uw7otM#k+-eA|52DAzGnYaUHk4U4<t"
        "sm#j(>Kd)rc31mY)}!lHvtI83Nlr3pI((Lb@fia%!cPaNjKl!eFbijaB;-(jV<Zxj=O?6eD?ql{vg-E4l*_rwa%YN=nP(rO_?qg)"
        "4(x(rQ8+WmFutNAuNpF=7`DC69=v+PJ>ndSWX@62<f&9$dIG+Q6rMm9l{SThLjeJyiD}fr2=4GS5U}?3!CIyxLh;KyiBznLhD)x%"
        "5heI2X~sa>Ro6H~NHHH;P?I|&IFB7GniaYo3Zhy*S_AC>TFUg&H#gNpZfuS!lJrf2#Xpi*s+~aaWrtG!DqEvKb`rE>@vXfS|9ke$"
        "*>~Tbr{BF`uj1ADi`DAotFyD!cQ2BcjIGYT{f4J;`r^A+13<)R-<(QyoLS5`>rj4haTVG<a(TlQ{`}_SfGje$dhfJh{pwdc@A%pX"
        "TxkrJ+wH7So`Q`6g3kx9$m?eJB;acT9IOUyv%Q!R+1NEJaM#bNG*FeHD=L_7U|2GMu=l9U-bthKLXPCE8}po`U#{Ji=1pXPN2jEt"
        "bpXSeK}<42gtdJx^LAuSz^Vs~RsbB?zvTjqB7n4|k2aZ)ZXxSu=5ocx<EeDn%!T@lb0k+*ZgmJ&&2%!6NI5sr!ZjLHb`aL~b*Wg-"
        "e9x0I7Oi8pKn^BdN+yhC=6Yb|R3JnrrBBHo2pHm<x{S*l$yR}SlBKDN(IS>Gps)H1o=VpcSmMF7fY5BU2c67wd_p@fHazj((;{Ci"
        "zF(ZZpiU<aRvLJ{C)JK~1_%<92U@bmBJ|H{8wLzIirJ5@Fh0rHx+n$aNp=vge8KwnY^Ou(k;uG5(zZ8g{BczZfo_$K4#ccyd4FT_"
        "u10@UQ2hTQyl+np)Pe9)TC(4mdEzo`E9wg)a-NRgx4<-EMD_h7dQbrD#)&2cDWK^W>)$QsO4S<+=Tw`3A(<qdlTKA{t+gcC4`gur"
        "c@EP$d%!gp`v4Ski|J6eCOozP*52beASFzCNZ^+hi@^uYaAVX7J9QAiRpf{E!T#{@=!1rugMeiMTO2B0R~c%amM=QYga<hDmSKNZ"
        "rzpL1@Nr&R`eU&Z3{5?GSotUL`iah|=lGuIoW`8vKgT<nr>E2ING<LE(jAqczi~#KYW<RIptA;UQTAfA|I$SzU>sRoM<hDbcrkFc"
        "PpbJn*)?fqO{}5kkjOtCQAk3$X=L`S{Ihg#+VMy{>bRQnAc)@^fUH0XbnOkyX_MET-zm$s<v}mQZhB~IFm9bA6jNdyww&Ww)TO56"
        "F=Uz$JunPgvu=W`cB|IokjcS8stn4bH6-)0JI{s%3uPS)4NcrtgEiA0#bC$;$EIqJP<Qp#)|_PtHD=Hml6mrRV>U+`C_C~|YA}rg"
        "3?a1}QG}C-VIQGM<WIU3Ya(r_S?#eYpm#(tqzsGyDn*rv^Q^le`?B<HNLs^@b@b}8M%UZtX--J7dTg113?QOIZ{DJrWWal4dsN$`"
        "*A%x!d0)t)$eW)!ezYesBo*>2gi0zu$I1vU*f&{ze9R|R&v>BqOU*o#&4XU{rxuwMxC}c&soTBv+A}Hl*rbOAb+*dsbp)1GJr~=a"
        "77eHjbYGFFaYMpYFO+V2b&@`4aD^zL5%_{bU3w)1l;q5eFhG&jqttMC#>7_j-fe_$FT6XFuG<-Wp~^6!SKW%ZmRoB7Q3g=v1Raa8"
        "eOQO$G=OT(UA>v)sg+uT1*)4sn?W2m$o50qq#9;Rb`A?L%(=ACb=jKKt})?oRHp00k{~W(wgWBHHr?n5H5oqL4}+K-;xHx0oge{9"
        "$qO8%%W~adM-NR8mMj4&r^M1TBUTMf?U?&~d*~HMT59Y$hNC7rY&zOq?KgO|Z`1+fD%;h5h%Uu<6^s?JomMaLEL=J=^$7OBb%L(V"
        "W>*}Oao5rkOBIc|7Ae~hDR^qn>C6Vj{u9?~WlKUNAbl$m6=<uQQzWaUfvKSG-;QZ|&Ba|-l?CR!zP<Kgp4J(?kq9VIf4ksB$I<=7"
        "v{lBjNPDu)yhqJ71fCl#Am3O=_f`O@N>sDK2tD~H;+PIMyGPe%M1C`YfXdhx2eXIAKJ~k%+Lc1qNzJ!KNW)3bs;G;L=-^le5%kP="
        "8xYk>(J4@*-i+(s7QsJ<ebJNrP!~(ve&{G4<nTXye}O?^!=)GE`<c3Q`y##{dLx|>sFwzh$qBMT_sAP>8tU?uMnhI_3<K>DMab0J"
        "0<NO}E*yTSD3l|NRw9C$StCT+mI*}!T!>ogx)f@1H_;-R_wn_gNLF$Je__Afal8z4fg+%<vy|RO)Gyo+ns)o$H8vuuyH;Z*6F?Y1"
        "t_pM<x_&lfmBCD-i(b88BD)@TCCHR|pbBlq1E(@5W*fuIz@#M$3_t8lH;5GYz>Zl?QZ034)`9iyr+Q`_rTfmyqx#O<@WokjneIqG"
        "qT!qxyY8n_fIpyhMtvE5gP9TF2k*_WQFuq<<_2=Ln4<U$Z`%slYJ|?g)Y+-i=hGln|8NHdj(YD%cQ013%@SMx-UyJE_hVavHC$cR"
        "AQW1SOlW=Yh=yV_dwz0@q}BO2$au%v9HnE#DlOK;u((cdKaGs0jk%CAGcp7vDKeTT%iSolR?-#0-Hqd$EcFL>DJvS|sFuqYyJbg0"
        "_oBDmkZ_1ovK3dBDhHKCIu^HXI$C82hiYG+X(`vQ76*eK)Ja_#<Qt~iy7CH#qAJ!{cMq!x3Cq%@;+9S>k5E*lSIbGIe~wZ<Pa%6*"
        "{Q*UE9g|xMNLyp}Tf2_Jd&;Lkx_d4NnZLG!Wp%Yb?s2fDKRIS~RryAa8C(^eL#GMT(bmLy0&xy_(*+#1x@tW!QkIsuK`#=EL1=KK"
        "&)FITW?gZf>>--~kBBZ7yF^i}M~;tm5RIo!$dbDyR(rem<SRifefmc>7JsYO4aRiu=}C}qnVw?`8bwJNN0E*Un-!i;(J<`n2Ue}*"
        "l|ZO^Nw$R*)aovea})CoTiM1+8R?4PC2-)=+m>_H<=Pue!<Zs4@aOYZv*(HhUcKM(rT$5P;iM{na?Glp`2B%{8I=W|;c<T!bFq}a"
        "$YA%Fhm+F}uwsOFH1R{4+On8|;sDG#t6e+;NnxPSbQSng5yT0;sA<&R4xoPSrVJ2CLFnGd1x$>zWa-ljWf3^0gBwOwnz?$7^jR3F"
        "X?-r6)~C27M~D-nlZ-%>#W*mF<n{BJr99DGp5ZApltF1Kzd8N>+XG0(fnS^-5@B{f0e*IRdU^l>B$IPw-2|3b3WZTMA64XeH^?7="
        "9aif=cWGcBm_B)@#?2Aco<M$_c26L9vPMndI#$jmP&`T0+9srlt?D)P+HlWg=Q#Y6xjByNSZ0o6coHvV=h%>svYu!T8fup=sUD<t"
        "6kdlY4lgqTLxt+OY<C{Dn_`cpBwgAMCoxK6j;c6sOYW04-Hk>UT<2JDeFnU$-s?rBdC5#uc{&Mvtjy>j(Xrq9k#7zC1G_$oqO0;5"
        ")^Yf|XHUYZa-X?Tg8Ksm7E{O$L9623GohbsN=7j~$qpQ3?1T$F`~Lh87D!wnV7iy96HYWQ&kkalSji#KV^svRwJrytKZu8lrFUvf"
        "|3@@*3aU~Q>WA#*8+NVJ{mU=IjMyrW`#;F~=-;tEI=?*s0sjq->GT?oC?(1t`YEv#PRD+-Nc*3!qmS?Y>+-EUJax5$h5wZGeDeeT"
        "`xmc==dRt<d7QP8Fm#)#BP>~Rv3{pz=K772nWGgh^#0vDJL*?O-WzAAt0q}vy1Q*e?X9>+DMEI~2S0)|_*tVP7u*k`hOGn>uWBAU"
        "gAq}W;CJ<qAHpt!7yIY4`aOUqjzOQq@OV%@pZK8w=T8GDGr}LTIyj$ssozyl)lZqGEX#Rv;e961$6wnY!%#oBcPh&*u*!%kXHi5e"
        "j3O+)M3G*kl$5KzsQLC1XQSE-Mn3-wP)h>@6aWAK2mk;8Apo`dk8j=u007w#000~S0047xV=rQ9W^ZF*Y-eR{ZDB8SV_{}yZ){{P"
        "aCwziZExE)5dIz@|AElHB!FMKZfl1Y0hTmwkv2)Q+1)S<gF%xgn+qjsBxT1f^51tyN|xf-iRJ>0NKt(5^|`wwNv?SYk_nSLZ%g5o"
        "$=OUBaqAe*C4;*{E1^77xre#s@x3W?$*lr6NRs66;P4=|CS#I&E;M%z98+0gtZ%mH(BZ)^oAnWX@fGrui7nyy@F485vr=h!M&NG_"
        "4;cQGe8W(zxNx5P(zQ2CX$7=4iqaOw)jRa>?90XF`Sk4a^Se)DcEXP0xdx}&pkMOt#rH{%B`CRu3{YK3OEgTX=7VBQi?O=vXGVf%"
        "t^lD@C89ydQepw(wlHWXR#>xyoaX|Pc>i(lDd6*cz}|A<U4{J{pl-?=yY8V^I34b>8TVol;0p^<MfYvK+>ef!FLz*)dxpYj%@L=<"
        "y|LiUE#&GehKUD`dfI+!tWfzp;)6?`Q}Bm<VAa*Zj5VOjbA+042g8dAtHEsyNJP%K_Xyt6wm;IHh2{blMoX|8XfAo-@!AS@V!6sm"
        "Cwl3?v-*92MTc6=3&u;Sd`IM;l3r{5K!nzQl*RcVMr(g!0s8206oI9Hp>Hv*xK?vAjyKF$HCH*;(Ll_kDGKfCE1r|>)jo%~6}U|K"
        "MP-#sBM@Tz$EY86SwRl(OJ(6<tUVEGS1(@EO*#5_2~+f3xHzPd8^+xV>^dx%MT48HzXCO1c$R^kqsA@%c_!MfD4kks3n^HjWNhzN"
        "`0@5BMiiQ%j=|Qkk0GI*NqxredhElmry(4;TW1+~s{|$cKhYRZ*dB%2m2Z_0kj+3!jJ-M*$D7+%#$s-BizH2f;1)=`82Y;*LYB4E"
        "-2iO%>R3l7P{e4J10%xDW_leIn{jcw;#NLu<6UW0p<x<%g^~IotIw)*uD!~Of|3obYuE?KGR`z#W43k5tt-l{eemH=h5EkKy1E|K"
        "#}~HpIFk^T58|pD^sD{Qg4l7xn?_dk)PYWi!Wt}bpMDQ!wi_~M;hRp-lbv<KSG+Yevk-?%sH5m6<f_ThwXQ6{E|sH7ALPK#t-)j6"
        "a9{3Fd4n4speYw7RT^<_M(-!g6kcUS-a+g76Q+$RIzs>O6m${5@Tmen6~08e${-`l5+qhKbo?5(xI+ddvvmuxNsMq(z#z*`_lY2t"
        "^NI>)6Y8I<ZvwcjYbGj9olz(7i%t}#@@jWD2Xp}6#a|PS(eC>x785qd|KyrQ%8dzt8oWgTsaZpLLnyKWa@y}&NnSscsU4i#Rj!Hp"
        "Z868#+Yy(B(%+Ab9dkJ^+`@QjLa~{Qh4|kR$q=bJho|WF{Ke1v+Z$Zo`I-q^_R(;ssM=L0*h*m?T0Pg?j3`8(^%X3+E`x`vk~b~Z"
        "e(~5}7(0EH*FL%lXP6Vlf_;mufzr+J?nTD!Et_LBGQ3yG*oFQt$4$1{lYNfc1bXAVsY@SqD_5Zrlzm|q)c3deZHzRaa-mBZnyW~="
        "k(H>$qY~7yYdPJ@iru-{I>0e6{E*1zQCb+IsjNDZDbzh0X+l|8f;LE&{qS*Qs!r04B0KLRj8*_(4M&}uz8;-=j+Y@tpxL;xtU23>"
        "Ek?x89bDNGp_vxkqg2FRFT_}Tr1&`ge0_;^%r<rXsq`C1TWp!2nUGqHtrd5D$5UtzP>wGF{l;S(g7aaV;HGDPPA|@Jj{6oS)B%l$"
        ";fbb#>IsvlJ@nnjIWW$b?_(Q%Z9E!Xu1D4ydxWct#$aLZxEeokxIN87hmc37_LwxHj3?_U7{{?3If}EC&nyju_{x<+poh~^Gh62M"
        "kW5!fNZE8s9ny4qM14Uo$~~<{T|@xSl@1HkG)Zb<=oeKyVQq8#Re%3?bB-<^)cbh8e%v}yD--GCgry1YxM8&4hc6fBqd_LW`&NHF"
        "&}Iep==eTq-9wGV(eYrTQ}c>`X|nnH{!Q(+YD~Xp$_3NaEf~lF`xe!|XUX<O*F?$AL*ekE$Ktnz!hrb?P)h>@6aWAK2mk;8ApqM@"
        "2_lXF006xJ001Qb0047xV=rQ9W^ZF*Y-eR{ZDB8Ta$#w1X>MmPUtei%X>?y-E^v8m%`pzbFbsv^y;ARBd8ZdZ-4GXm4WY<&lVV8X"
        "%1M#9JpqZ0?G2y)hlmF;&vkb?o<VZ;7KnY>L)c=3Y9v|m0_Ia&1qf#4E%^rLFJFwRXoe{&KL$kXyS@vi2_^~36c?Wo4ZqN!)4jdP"
        "(-)Ams#RF2^3AbnDXw)**Vm`v9L4AyjBvp}1h0Mf22e`_1QY-O00;m803iTQ_x$jd1^@s*6aWAy0001Ua$_%IX=ZO@VQgn*ZEayM"
        "baG*7ZfS03FJpCbVRU6=Uvp-3E^v8`7~5{!HuOC}{((~;DhIB+3>yZx^V7Bheb@!quou@5Xo-r6N~AziPTU~>zH@kyNJ-8npf!1J"
        "JUn+M%d+37DNn2-b>B4aIw>1s<QFFRS8BKvgo=ukvh5mX*gMwA@<jgn_&2Ggf`7dt9py^DH;<AWsb&oqOmDI*yS=%+;cX|CAuTnh"
        "+nZX+7KqKM;YZ>ze_D*k9eFKF+B~qfrJITxx{-B_@(Nv>z|lxr6=koeVO4?3u8B7E*s!AJ;(@77abd0sOfD8nJTt{>4kA4TQ%f?8"
        "qFlf+ON!Op6cPAU_D(juyaditikd(4iWLrCw6bDwuZq>QZ%h%`icZOzH%z}OTY-x-H&R_<?ZUN+2(4HtRfXoiy@9hx2RI5^1nUNo"
        "x$aoGCTQ%U;=qO6>p`R{N(%9_+TPv}_`vl)OQHLg5!y5aH2um1`Sgn>To5>qsSiX*qA5fa2wxM$-f37GKoSXls2ll6DkZxPuVW(#"
        "<tq88of2d>kAghaZsm9y-g2XX9<loj@2*UC*k@LEjVniz9UMaiCIJo6_GQsh!E2^XIBy;!)-s!c38Oimn#HTageQVUEID^*tL<xt"
        "4E(Phf*-F*%rGKYqZ!G6mx8Sc{_lDGkL*p?TU$N9I0z=Vb<CP@!&>V<bi&5&^5E}?d2h)NAY*?RkthfFklk`deE3218}@{z0HVzU"
        "GkNAg*$P7<Psy#s?9H1u<de`w^(ChAXYmZo`?q`qy*<2q1~?Lagq+zDM}hUnqTKe#KsyG2EInnFZiBfi(DHgFEol0DcL(RJ$#A7@"
        "xbMq)PHppS4i2uLV?yLz<tc1<?_7}YGY`!U*JVvOu(%f<l^y0>SEFrE1fwoU3k_nFxPH~{X<Th+*D+BI!C$3OzsKm>U<goAvVvDO"
        "@(^e%Pa^6-z8Q-G`VPpk960PGLZJ-89sZ$B&wf=(syzGXidrd8SX+6;gyH5w$`geaRFSHOCiIYW(M{%+jZ*)4JSGEGp-Vz|WTC2I"
        "H@2k>%Xo3V&mvQ079KS`V05>MVX<*D_S8XTX_=+9E)%q6F@>*$TkowH6^yEfb7HH-lw)C{S~Dx|c?i30Qv4t*LKtk5diyfFsW05y"
        "NgZE0xSE({J{(Na^m!={Aquu+KgZygKgB_|CQl*w(Hgi7nFRf`=1&1?sDk2#yVbH>jMk^2O3rB(vTvzG`#=v>I%Rqpw<~YGmozRX"
        "Ap~K%V8KA_r7LE7HKtf*2M2J^2&vc^_knMmIdEzU5f5?hed7iox3<F8p*^41al2Zly8G>&|FyHK-rf7{5y#v@LyQa{T9$p=H#S3e"
        "-w)EB3T;@YceYo0S!dc544U(0_rtf}dZ4CfHUlb#vrmB&S}Xu&w+BEaLN#O6?*2#EMZ?n{f)Uk=_kIY(8OmLR94}!c`~anfm0vmu"
        "&LHRSi0(Vg2u&c%VS<EN%4so9Pf#Ddl%L2ww$;%Z0TLS{z=Sl;0bv6kh^phfGJX<ZC_Z~I9Qz7_e<*@oLxlj3_Z>{-42KS{+?{<Z"
        "Y)ZzPykR1@XRpY6l1KX9C8~zm#~p#Q-b6TDeQ+t^+Ri?%Ee+toMqUWq{(s$4b_QxCR6F7hbER{*W=&h(T+$&g)VxB9<`PXEs3m$l"
        "6qBIobbib9rEQr}yv#-(Th9%LVDyzJ9P$W7T072+j-2V}=U*ec>kp26wRd;%HE8+8lKYT#uwVmxsklc7y!$qw0L7PBc$Yx(XmBEy"
        "18arwT2bXGz%Z1+(?{N5k4@Vq1v)5nCDb2=%~Js3nI$iSrS_RL!^niwmS|B0qEb2|;i*xYe3f<}P{wu~mD}~EDE#$`BBfMRXzcia"
        "1qM%)r|^X1NXsuriGverjnuZYrp0{-trM?cB(cdd=<RVE=ekWqvc4ka43Ep>t3pPKsf-YZOPYAxf+neurHsH5lxD?{8BtxuFBu|S"
        "@PY<z5u-2|CnLj^R`?(D*T^+P9wyHB$z{{T4mG$)-5rV@pvO&25|=<AinlQtwit{__(}UcHi!nx<>c=cQK^c%Fn)0-#&7J#_}O=<"
        "Ys+!yCfOUo$1dw?ZiMoE+x<lFsDUrqB<bc?3^K9)Zf5(a(T~drANBlk8R6shAFU&bVG;<7SUOGwk+ZCsF+xLh?IyF%D)uIXp6`A3"
        "@n8)Y+MpffCap2$4YNZrxW@8Knxvog9MkR1zfem91QY-O00;m803iT{t-9qN0RRBy0RR9g0001Ua$_%IX=ZO@VQgn*ZEayMbaG*7"
        "ZfS03FJx(RbZKmCVRUJ4ZZ2?nZIQ7~12GJS_ml1&tW2#0R|cj-VuFqg&>f{Y=Nd@j!*L|~_V{`Q6{(qQ$MOI9k1=L<ci#g|5-8gU"
        "(F!u)_k<i0W_f^20wGPrf&&l(<5vx=@%{*d22S8@yX%ZGZPPYOCkU}S%(-Vuun(S`jU%RHyW*_(6o>Or*rJun_(JlEb~rvW6VET("
        "W<akBMM6|6#e0ub%Bn~m6ikybNa+`;Zil%L-+-WX2jr`_Yq)*<SLoXY^eFna&J6v$cnmHG*ciq<8?Z^Xe3p%1QcF&xVDs>+1CG*E"
        "MQhGc%@XpsI5E#<8wy_CZ?!a>#MOUu1Lg^R$T|l;s&4Fq8hp9?#h1VyJt>Cz7jZ7A=pRqW9vQR4?hLVRn;%e10|XQR000O8001EX"
        "Y;1#b7!Cjcvo8PuDF6Tfb8=%ZVrgb?V_|G(Wo>O?FLZKYX>Mt5XD@AKZEtdUUvP47Vr4FHd9@l{Z`?NWJwW~gt3I6MtQFhMr3bfL"
        "z-gPq0Zkh;x!!|e2waJ~=2<H#B4vA9<iFp{kQAwpUE94o0f|@Q%y7PEI8qcvAGW+KS>0&4mA`W>>+0oizh1m!O<S+Tf>pI<`dZ*^"
        "(dD?zKD_<NbjxKW+VuIe=g;2Ng;3|Lt-Gqo083V^ZdrXR+LAZfrsd7GO4&ysQgW?D%UYoUTQS^t3&^GV3*$ihmbQQXj2RrZ66?AZ"
        "Y{m1NJKh!xw(2x1YN3d$yySAr^18~qwiT6T-~aQ63w9&gN|a38i@ZZ>z+n_c$fn#jbqm7Ctq5=HHoqR;rB%f^>`_e~r|YgF9q<w%"
        "-Xfjn&wvr7*n8C1PxY@CKY?O+8v`^brYl8q4idpth&9VHSxKE`u@dDP<l#!3!$zsWS+JH@H|I=(FiUp&+F|0Qx<<5dlJ*o5zb4m&"
        "$i7{&lK=%TFq0DOH(qw)$F{B8I6@W;__k9Tbj+GsNl?B>qktwdm?D3`3I3T`Nn7!$ldVu_<06aS@=}R03EF8%8;({~&P!QsVDT(H"
        "VeCtAhvu5&EjyZnFYA_P7qAZcde2+F72r$pP+Vguv2FA&MiNHFsjRZK;JO1|7hs25nTsVY+zGk4)~R^}OM2Icr6IB|Yp%aK3*;G#"
        "Mhv}q3`4(*wpKCiSa&+$#qjvQ;0Sm?$4INXg@85=W>vOeN*BwM^aN48FG6f#R9RQNMHf+_$3a|bG{@VtG#=gmFtCQ7I9Y_|8Qi@I"
        "_=A}5+R8hRC6cfgW}{TsjxNcZfD?pnVnPl1hKOIw4d^eU6qQjb0*IJW$$}j%z@tl<@6JK)76gcpnP?JR?D#uY$Xr{IA&3kbB^StH"
        "e>l8o-IZnZIZ(|ssy<pK91Ry+NCLAVlIY`$7#&BCMO-fJeioJ&ywRcnmM^cyj4qvQ0c&7YR)H%na2LovC`l=mHrh#XsmRQdij!oN"
        "E`YXFkeW5GauFM_g-=ik<5fZA(^fU5)Nz!8f=-jm<I^kB*^nGw(I>RizoW&>(#`&*+x<)R?9%7+DQLs7Fl0s_P<vVl6$CTdj0jU~"
        "1!W3wtmZaCXR2Enox}m<Bp|(*lmVwhbiQO}6Kmg;H;t%@m<W2PFWH6KZ6UF=GKc&qH`Nv^orLNn?jY1^CbjXkR&2ooO<m?Iu)D77"
        ")DuFD2~)n8Y!eCw{w=B;ezlng0*(A!S7KOW0?+V}p?8cd;DN@z{;};GZO<@xc;m7txv{?k!Hg05N8q~@2=0EkgO3zyU)&Z4Blzb!"
        ";KlF5?gX?TkcmJNFp~>Qo6Z7%k4#WT=QcV6_l(f^ut2qNCVUxmK!Ow=A{ozUU9bYa^_38)XF2mM1ElJ<vDZ!2ZAA$Rn<%x%PUdN@"
        "cA7TUqEdBhZsQ5kXP=vgfOMlM(L&a>+E+rL4?=Lc7Dm?^`YGsL%q?<hP>DrD<UEXt_ue`XhODdPr%pfz2}M>>vnpYYL(%+()hl#C"
        "CJP7~sdo@V5Hp1+x<<L47gPa}1tm$PLw7H%1c|E%$oj~~@lnc)BKFXuAUD#zSXxPXV;s2$;93~enF3bgp?eXo*KW_SRd9bl*0Dnm"
        "t>L+%GCQmoUKQB9_hg4a5c^sT08)CCnz|i!qzFz=No{+8hAD;C7q(&gXjExn6R&|y(`8$+t=nB}I;5E@ZYY>#2bE$=!M=ci%e!Jw"
        "FRiMO#He+H(@C8J=Y>(=aulBiS&8=c5X8C_V*e`HcmHxD_)Yc*HVE!8FIJVv&Y}-Eqbbx#3QEne!pUk~JEaYLX{4YrS3^=@AJ4ZV"
        "w+NL!;B|`Z0A_FljgmBeCDbPU4%TK>8ScyPvqvyKr|Eerdn5>a8Oq~YmcD<5%2jubDar~DE;_%)Gr|$}@1jx5GwSKYG8!JOp;nhr"
        "eTf8QJ+V9$$FC9}5170RQEWs@QQ0jLj0e26J0*-_9Cl*gveSdL@<Z1&<&J@i88$&Lt=WSbM6Qt|Xi5+uoGcLezJ>yCh~xY1V1j^c"
        "_ejjc4LA-V`mX!%D8lXEmqMd?g&oLcW2)qVmSfav5WJDy;i#+e2C^XPP$U{thG|Dcm@v_APxr%423#{9BP7nMgWY=LVx1kXBoFGb"
        "#P$--9I`3|cBR;IR$xm!R?js_Sj@dzjl*M;E$;2HLtUO+4HTK!{M<{&t6kjKhse79HKW@@_BkxtML%xxjv^Gb2H>gGI4B@x_wMEK"
        "<q%C6rW=fVT1_tszWrBB27zd7mxF*@Aj^X1*Gn@YIqU#tgDc+)jpPqRGzuVt#6G4h`&99kGZin$S87+~*KJ*4=P^k7BREWEI-}%g"
        "(;L=wHN>&gy5&rfww<5mtvSl#3@(M7vCeXcRGgYvkP<pLp@PG%s(7QWYdv_@<+)J!g!Jjo5xdbT7;_D(4Z7})h$*p+PbddkpPAF7"
        "Z+mc_L(z}nmFYUe=vW4)8h5!AhjtjHv$+jA-E>A&GOfwXiT46q6_C^4<xHticuv%Yk*g_vJ5~kqW~UU@ZInRwqyfXj47QC@!hr)Q"
        "HW-9!9#oe4SjFzzlWCx@C@8flQ&PAYblw7RL6iqWNq6;L;f~God;Srj)$uq4f1sd%18xM|MvzA{JO1)T^yp~nxSywlXV{r)VUH3x"
        "cz3IrQz6QYYMckQIe%5xWo%+msx{8|1leFRyP0bHSBNc$wElow8T9>n#fui*$UQy)Bzo-=N+C3)xHw5+hiyDJ>OgZN=)x*#;hb#<"
        "iihDD(f(8r>`{?*_azI%Y_Ao<U~!I*Q?@<E3BaFhqjyrliD>h(dYr9-t)6JI5c6Tn;f2;SSkoM83*R7>qHV^tEdY7gP?%pO)MiRp"
        "O^Uap=E$^Tv=BtoeCongPtlqW`>8n-Zjuz25?iWmTpQfO3GN|t_GD&l>e@U=v&7aW<tu1wQXJ<b>^pYy-!g<PkZ<6;K5&eJMQxU="
        "@VGCUIsZc6=!uO6B;!=QH!1t+1_A6z$jQ|E{%;lh5zQ1{k8P-E4l-&Y@fyuz9uR8^_=LHPLwRsEW5s{ed<L%|=%88bN=&Y!mjP&L"
        "-QhSfdVlfZ!+3rlQ7$$OA;1LZ)Hy(az(3<9>hhpbrKzbXt(U1WSg$B#X4=OO+w}>f-cjv<jb2#M*tNz6H~s@MTdduCEShKbx>lbj"
        "@`R2L5dGBkz_@Qe2@IJbnc+-!%nnV=(pfj3?Bdh;SmJ4@kq(+r2TwT1joPjQjLZtzJo(%>-r#`|c=Mz){Dis^d0iDmb$SqD$L^ps"
        "Ml%&0V}pK-m(BSLmd%gxGGA-*_i0X@u;)@X+SmOa0|m2z%)GNka7>_MNdJXKhHjOaFEms}-*1Qluqe&~w7~rStcRp0V_TDJ>{%Hs"
        "R+i&a;NyzJpE%l$lWmS?{?T~b9BfE}(S~2L?}1t1s=lj~^H7|fZTSwQ!tt5FiNuzX-PkX`y!!^o_4<H*XU$j8KjEM<9nwBLG;o)R"
        "f`j6^NhMS@U`5LuG7D&2W#FG1oGZ0rE$=f()f;^crGcK}j0rz!Fs~GKeFK5BfDkDPoQL25$}$s0(OU`q2>0}DlpULGEI}sjd9KSH"
        "$S>ds4gE`3q1d1%jvxB)ZBs&HOJDg|y$AUS^Usjc#5Ieu3R+o=*boY5po3K+G!$zuC46>@pW_VS@yY2|rzc;ZO`yH*hMV?}=?(iA"
        "fPeCOf*dH<PMHtR!DI!sTo&2HtScOw@^<H%KyeRwkWHj`0kR#Bv~^4c+q3J(QHIF3gD;b&JWYA&iVb;Wohkf64IDGGmdE$g{Ro#N"
        "2OeYHHS}KI`x!f8acs7KVJD=);xRi-*vprbg(SH=C#0*v`;Yi2u*EqT1Nu{owm_o<;R0M*wLoM^^G;vkkiLJka9`IQ(XS0kOmA$)"
        "ud~3Ze}U7c9j(L3fGWl}P$BAumB>)btktA2bw^)w?GPw+aL5$0>{{VN1YdF7(IhT>f&E6V>ztQ=7uzjA{_*S${tKI*5x}cA`0oI~"
        "*+T%YrdybmSA_hJkMm2f!lloH0XeDK4a9NuO%IZO8g6b~+;{+tOskYfg-&40Ox!~Om-q@-Ik&{;pj`UyCIC$ayb=&_f~P&dr*TKw"
        "63b_9os!ROm~Wg6uU{eWxA@;zXJ5bmn`!S&S0cj>-Msvlt!YbS@svyvitF7ic#nfT^2?JeIHRuc53+rEYHm->E%ucw;({;Whn*0Z"
        "Xu$t3PtUI&7~qF&-~>ODRo-Hy?7zyW@6<V|{St%dDi|8O@1k&HviYb_z<%a_!D9O@%z|_!ip=J|1n~sEyT^$>u{}O_TS&}B(`swU"
        "LLwT{g$I8<JtLREPm-wfQ+HBd$GyKtiTOTch&bwyhqwI^jM7YOzJKZ^b0Up!KzIAcsWTsT;T3u3hdz$>*#~I9F@kPTLwdFO9grMb"
        "d+>%*d$i&6eb~@AbiABmKY(WRhhOZ_k9}Y!0S6HQMM`J*jq?`2)4^sN9h+Q>s^<C3Yzi;@fR}c<bcbZB_2!&Bk3ueSc{Rh@`0zA2"
        ")Vqok4$bC1p0i*vdIq*1Q%d`po}ssbP`w${)Uf!dyl!X=<U0ZC`&{nn-;&!GP^f7$G57=)mU=g+705E05B`LYLD0;-^0U|+43=S0"
        "ot|FkHG%uul-drJH8{eD?8EQnw%gh{1#XjIehW&>5#0n?A1``2UY3Y(>748;P~Xz2k5Upq(fcc;?yKE@08mQ<1QY-O00;m803iSx"
        "W4xPK0RRA$0ssIf0001Ua$_%IX=ZO@VQgn*ZEayMbaG*7ZfS03FK=RMWn*+{c4c!eaCvP}v2KGf5Z#gZhpQ*M0*j;@RXQ|ErjAjT"
        "2t_WKV9A`3ZKUw`g_xv4tvBC$_wL?3LdYRo<sh`Wpsb)1zHCy1dVuYY!`5gmys_XKReJ{pOZg7kI0tR-JfQIb0&x)vIUyvD;z+ff"
        "u^z_8vRVE(BCmTay)q~?2nYMAA&x3phvDbaj`C-T-kNTpaRlSj=AyjIXN`kIXKVsmd`tbOIvO!no)uQ|nu9_Q8%S4K7Do)yj|Djo"
        ">Z+axaI91dFB^D+(Q}Z0)YenY_nnq>X1hjo@|3R&HsLcAnr;xA^`~crauSY0_i|^gv6Sqtoh<!$^PmQ&&*UpP79g-1<FHax!y}fU"
        "dZQGau*cfd{lAUTbe=oy@$8>*fdgfWMWpX7-VM9Hr3_vn3xpZm^Z!El4T9I&2u~Te@@AB#s{|HDNmm8C2lT{Z%Xl`=pD<+f_5)B$"
        "0|XQR000O8001EXE#n{Uun+(Mmp=diC;$Keb8=%ZVrgb?V_|G(Wo>O?FLZKYX>Mt5XD@GNW^8G0WnXh<bS`jtwOe~{+_(|{KR~_%"
        "E1)>ZS<8>!T?;o`AWa;!z{Pj)<ysWmg}{}#yQ=l79=7*_eD}@_Nr{xW>veJ!a2Jb@;c)nyC$*mE{k80hL?rBo^7ak;`TfVUcq`(&"
        "s!WTPZADgzx?$_GX6;t6oVU=Mun%R%SF9*ou`0{ECVctg<%>57uUb)`mPNK>1C569rs;A)^Q*KdV0u-Hw&p4R*_2%!3m8-j#_xFw"
        "bTa|;Zc29h!;Ix+A~IHWt1OLqo0i3l#n8mWqli0vXULhztFq=9<83PnYK=K?0AxJRL*7yx(8APe*S~Cev(3^~|Nft*Ec%z47fG4F"
        "e6g;}oK?KVkxcdd1h1;6>(WFw<KN#Ji?wAP>v0WV3t+7&L6)i;o1(H3#dJtWQ-^`~B3h@JK-x5Pm9EQ}XB&~{JWP1Y!??^c4zu)}"
        "TT$_}eydtt(@6elu?edZM7z@y{g*EoJgY9vl1Pt<Hg(xm%}n-{Rh#DNucDUCQO=8WEt*z!cNK6XB1tyN1h0ui4UI@4z+Jq2k%%?J"
        "Jw-{nfw?}4IJlA&WD}d~6AGIxzx?idKakxIX}hH}kds1`q3_izF94*mrmY3f6(xKSy79Iv?pRu|6aeaHJYOaJN(~L+e~Eu_e)%ms"
        "VfarlV=K=K3=#uVq1>HM1QkDk<*qLDfUS5`TQYVUR~-mb2>+gEozSGyHA|bcXj)#xs%s`u6MiHhm^`ReXjGVp7Vzx{ArRJLaZ*dt"
        "xOGtZbSHLK16LrGJRm_q`wWGYqp;*!_}2Kr=L4%`Y2#XD;aI?;y*wm`uF8aiD7E`fQMQkUnugOLg=*GqG#Dubd$KlTn5%bSuNCU="
        "x+?&<x+gAkKw`Q|v$Wl*wiDggFA)KgU_R%bsfLj@p~TPqg@+oVHeHt`Dghx~Ef@e;n=-o>`p|JqnBPGcC_@dlubF?WQGDVN&Lk@D"
        "=v56~rv=!^xg>nX*51I}=_lIW;+suYu6PzTt*E|PUY&oReD?Z7qWK|F20)lc!_!8v_n^+#)OlUkW$mxMw}T^*v;GVxvhLulH>|uD"
        "^+OGZcEvss;m=;s3wk34HhnLIt+gr7PCfQYT3J9o=$ScQ2|+c%1fGL61#3HJp9V$?XuV?9NPob3g45V^W!Go|+Hl~68LJRigpCjz"
        "l4ziBP5=>Aa6td<QaB|rjy#clN_BcW?hn&vXoP&Z<1^-Q=|Y>(a0Xt`Z)%s7Jh6A=^)U4eh3*?eI@SoyfFA@iV_t9DX@qxW;aiSC"
        "OBB_`20Eox+K3k4<rLo{SMAzuS*X#{dE#)#$hb?mMHL%t(h<)xKR|gJBhn`7;{)gv89;8aoLM54!%@(_lGcp%!j%z6Etd$ZR^ts$"
        "f};~130A5rN_DKl=57pq(K1|EjZ16qrlmmA4$Co|4(>EUz%@u3yt7CQ=mhrGL0A(;*?i6pnVU<`P+~2Hg*S5Wjkg5R?DP5t)`&MX"
        "N6B`8wjLj1nB@f_>3-CT$JWP|FzNED@ii0Rgdip@+W94TV^KiJ2T+}RUAsR0Q-5x}z7w@e!Og%zGk`q~<wni+-Fi23KxQ`Ch;<0V"
        "c5koWy!+P^dnNvl*(sc@G;+Q552dwdzf)!sb|bCxz7Z2^{SWks(omary5%*hEfY=5D+tfQ@jwt!ix^aE7g{D7)<OyTw6N`P(WvQW"
        "?R9}Opz#Hc*MRk5pNqh0z>YL9`TygFZC1uOq_ah%)A+t}UN#Oe(v&cm9CR-~@*ZU5zXFwmO#pc)pald4O&<Wmi=A&MgrPHvc~Tji"
        "B?&C>eQ9ZqCMU^ELxMdZOCbp=@B}m17Nhta`XEWfNG(k$&tq$_^&$xeZI>I|Bj>Ssd0NyUoPH8O-q!}7c09q$p{?18g1J#;hV!(c"
        "#QBPmal`~zCS5vMVE3088c~{je<3p*N|$SKpF&vDXWF4REH>T8V&q_!v=y`I*b@?ukoKoJ7?3m6s<XDd69wwzS+AWyxLBu~esiH="
        "t-j6^O-^Rwkm!t_>ybtY-NF42yQIm0(FFERzW7X?-DKZ((5XVrW1~n%@MtLU#P>_B)Tcg<7M4d_dp&Vro|tXG=C9WvKyMOz$`A`*"
        "dfju_qnuY!R?3Ra+=I9j+8#u@*|trD^&vAY`D(miyqBm`1`ZXkMjYcU7oG_PZ)t&TFOW>ahaqP}%MCz7bDxCb&$#j!h5A$&RdeF5"
        "CI>I!*q@ZzlWpAb$wVPHq+|h7pgm!`qGj~ha}mTech)ZpSpGGwED_E2Y@CuD$Uw|P|1vQQD2>YwON%3z5&}&KjItbc?xfAM70@gg"
        "uEKU|hJana%YJV;)<YU6WtXD2<i$q#X7CXN=+pqQ?3}%x>vdjBS<Bf6HT3TV`9VW1K>Fw{Yh{Qe#n6rqCz*$BB&@Du4Fry!S~~Ji"
        "ef`qMXiwV$cLqBH$il%rl=a=*BLk6uV~d3}4OF*7hb?N`3t;df+rvAt11m{F$5z7>KF6#p?y$@?!H&kcVXTK$bQNs07-pzck5XJ>"
        "7TbV23ZG&p`M!lfTCpw%S0S~Hi8;BqiYPSAh$hI%d^R^ABL_0kMM+*{!t-<Yh))$_<vcRCDbH5!GZ+Q#<Q^iU^x!hdft9jbm4!il"
        "`x1m6u1X7yE-my!jRM4SP>a61uR1Jkh@nP02FcSJiVqU55m}vygbqUP?Jw`%D6~tXrEu3+z%!O=CY2Buf1`8@h7OsQz<MW6Mpv1Y"
        "4WycsUO`kxRs;<4$&9TKLo!!V{ONu5QRiHRE82=0xmt*e`288SJj2p81c^B7JUpMVbJbUyLFbY9gbmf?Vx(;aYIshX{lH}21y||?"
        "KI~f%Kj<ke<N{Y}p;rdIWe~QduLLqBU3P8NwOTSh&2tRnO(TG?93A2rVA6vg;ifFabUYvtJ0ZZyd?}u^fdm>f(l=-XuFVbzyajYY"
        "%&stagW)++z)QU36v!~7VZly65}0#_3Ax-;nDLd!8dQVRi}Uj!gxtgr98>A+*{iv^*P*_OCIy1q6S;zGHbLgq+#EoagUp1D0YhJ-"
        "b5L@WoFsJ6t^G}lp{XDh!pkL?$r{MV>U$WQC>p(h^j(RETcT<u?(|FzgPePq8p$>qq&hX!pv}VWXY>B6jmq^pgWwre)lmwe9qa0b"
        "xPSKTL(U(`;Wu+QL)9j#GTuUidL5A8H}N>`@-CCX`?spmy2zVfiqr2@#{;i(2!-V`x-rhu3PScepI?Oi!eB8XaWCrK{8Epjn)tX)"
        "h-7~8N4;6fiX#kp=cqd~ZKKK!bTwuhhG2b2Zg%uhpp2DuSAP$u?rTAJ@Mt6t8%%y9vJX^_!QK$gdo7)S=<gbowFOy4jewjuQ5JAg"
        "cyjatc#SAM2Xyx*Hd6;>8*1}tW`;y(h9m>_nq3T3!JJxWhjHlnF$EhPg=*<x;MKyvjTv{6?%@C(@d8Y#qN4#o%VCR&6X5Xt;MzBU"
        "0U%e1ylQuHfit8G-k+)17#662n=?sw&|1f>nZ>OEC?ZE!X!=r%S}=E_R=h2yTcxKjyGAvjTeED+c!q(~4l5=t>;yk{Xwx;1@j@)#"
        "mL<?D!~?ti<tDm$^Whq$b$k8uFCVU>_qRWOaMXOBb?s->B^I(GVbmg?s6s3);SGZiA7AMzEeybM<ogHH{fprea-_@XR#BY|c7}9O"
        "hj7#qzDaogPwVb2tLK(#=o&?oh?wu@6vhC50({y0J<l2~QM`LH7VIkz&aWnP@HJvQIF@0ZLm$T)N!RoU_WcMxCnx)^od~)kKIxXK"
        "+JQ?K75?Fm_c5M7Db`)>FPo^+ib3MDv5jYf<kIUMB?jAb(5c7PpMpG4(Y#R$U{)%R{+f0f1a{VOOaJ)jpdcXHcHp~(%IA&x!xepX"
        "K^1)acJLg-0yOICZ&`1+(&Y$UZ;+ka6t7EA{EOvK2{eqCSpZXIkAqh%`Z42+2?9s$#&V;cLl7F|H=Z=Pg*kM$I+$(Cv73RkA%ngG"
        "VGVZNz}YMr_i@(5gV%Bs9{a`7<H)-B;k!Rue8*gN8WHfsh>~GfhC1VodH96Ap~Pj^hipK}EA`WhLF}X^VY{mY)nhUJH0!QL(h6iC"
        "ZLI{qi*k_zZYh0j>^v$fH}w6(@NI`iISDIMa(&q17$?K8hlrLuwHhRq3cF>@G^!%m)jakWS%<TWe&ft)L>Us43wyvoSdRg^ljZfZ"
        "0!w}8TtTh>lm*hy)Dl7VYBMsJrqLFi>-S>mqe9s!s$+s*f@T`mWkgY`Fp-E4#v0T=B=cqn>!_!F(P`}wF{o7<6HoT^#dwjc$+A{S"
        ";IQCcNe<OwBPm<L33*kCBJn46i3J_@0?ojqykVt3>5D~4b{SaJB1keLowQ?S{$%1#L}gtBBR`WX`{JNarX~F+OI<gP6dIQpf|>`|"
        "Liu3T9fnFF{HN+bXh7OB!k~YAQIHE>P$A3Aoix1UImJ~s+1QA8T@~sl>@D;hQRI0}f#)lVJBJH9eP-q?>`aR~wCQ6CJ0@T3g!z)x"
        "nF$OCqgFh=gX}Bu$C&Sr>&dc^ld??>zyFp*R4Qx+-aJW#@_$u`uS0fYq&<|?P_J=D%L(iUaw?a_;;ZY_HbF>Vs7bBb;A;I`cX8N0"
        "!4GMJ-$0%f)06D!SjS0;v`!9+BJA2UYeLGgaRMdUk(Ju?`S<=b;Q%ag;slR}v8Wj*UM=S7%WvS`R*ZEG`7eIwF{io*mWi0}bo#H@"
        "MF0pf>j#vmS_K7)T`o-9a8#~HoRs>O-!xm+4W>ggsbk?~%6D58mXm&W!@N<>XbR@g7xTl%3^Qj)<TR?-<$t-kMUCrmIJ0f)DCwA-"
        "l@%w|<-;`n^p5*hl7{MR$qsOw8Ed4fw7v0}jG`~wMwMD=amGWfBcKAV`*g?iw9rWLU>>Q$$mqFq-6NuHe9UX2T@i2V5<kkBIx!IZ"
        "!jmzaEIJZM9Z?)RwL(3MAqh`+wxt#ctFrP(;u;66$cl|=Vut>5#pyPnes2^xQ7q>pCaCwwK(%KFClM2SZ_2RPZkEQUmt}Kj<j|8p"
        "LhvT*(@l*^%CF`w+Rp|M(V#R992kC|gCB3CE6L>mF3hr29@mf4C<9Y`d2;fH^NA(U6^o?~b78UZ)wE6l)sJ?#=SX;+zv7YW(C>99"
        "qc)RiTnD^__wQo#5sbbh7}3$qBL??FaHy@jAZR7qM+4Ew-nG1@PG<ivRmP92s-vHM)g9~n_Vi(*zNhO@323^@V=6xS8HsAI<V@{w"
        ">)7;cMy0Cbt0k)Rd!)S?RQR_vGj79hta&-`nUG5k2F#xi@h9PZ9JH03#zD|b;&KGRW6-u2Sx(-+9ANtLc%bUx7U*X$lYmHFHa>u@"
        "?QQ$jK6h>3OpZek-91GLP@FdIn?4nR{X>}P^v~wcSMZQJoZ`Lx@xxC)Tz|ZN=Y9FFW}b`!jbi!O7g?i3mPC+Bl5i|>SijL3kZbMP"
        "+^lDmO=;2EH(CN`Vb!>gCK4kryxO&*v40bF$O}L1`R&xhy*!p}>-#5H*Ta6Z#n)Xy0W=)<3jF^Te_#;RT>%_Wmg+u_=(wL`eZ>XV"
        "vo5a={_sf&$B}_fYeZuaL0<4bP)h>@6aWAK2mk;8Apo>CALbJX003qj001KZ0047xV=rQ9W^ZF*Y-eR{ZDB8Ta$#w1X>MmPaBpmB"
        "V|gxcd9@i$Z{s%h9w7gLRWH)iqiI^ShwS(;i%o$6(l%&kb`ORjFczJ#BT*_zC6meezwdjbNRhN`r$ASiSmOKs=1Z1kSD*WaZ@4B~"
        "+VbX!^c}oD5h`j@GtFen1y`DHNY`uL@-Ix1JJ}7rBHK<9(P?(ub)U%ZKYb{(EIU6tKjUrRNli9g|Kwj<s_%m{-O0^;DFmU42!RJ%"
        "*|p@BZ@Ud`?pWK>QcDV~#a-F5wv$g)FS}btMglkZa^3y*?wSim<pQ{gEx#Kivv8HeM%w3RHQRzVTRJq_pCI6tH%#Tl<@p(bXVbN9"
        "C&-dK+jo4j2PPHoM0QCssx=cDUaF>(w6d0}9xk#)057Td1i%+Rh97Cu-&6Sd{deKltz_&As~$U9D>(ch2R3uuH*Dq*=qdc3{n2ru"
        "odi4k$7zPDjGxe1qldX5d8VG)mTAd1*#g9y-kdG)8T|}I1Z~+RdCnRMEWcx~e*mOMD%+v2lx96j|HIA(1VCC2Ol-QENu*yh`?J_h"
        "#QE0Avs^|J1OiM#tiXi)PPAbEY>g5f7iA*hOtG@e*0Fc-dG!h-wX7NI@kef>pjgCEnZ*a{_D6bbqN<vNygZ;?if7dWaQh=TPJFp?"
        "R;)S!awLkJR$@{hCKQ6~EAbO;R1|k!-e>RClYy~|qf;_HNZ}QmGV}pB*Hp70MNNw3_RoT}z^_^U(h26m)ERj!Tjd?oxzn_;AXA`0"
        "vBJ=NJ{bIq#5TJi7C|J%$iHbbuq!D$nP(rxC(%6$;-q~&mQKs94ETq_xq4ENua-wj3V~b-(QLk3B%AU_W&0JxCkBpq76j|1tM3mN"
        "ge#(k9?L1K4@n!9r~*;Yh**qL0fTB4!5tOzKS1}*&GnVHArXw#icq3hPbI_xsNuq*cKiZa^;6ICPz+Ok3oij0zze#vr)%<-TpSh4"
        "7z$xYJ19+{?NA`0xaJR7+%&smGL#~ph#97e3#F;p#G=_IP%KCeBo<`bbW|6`glgNVX_X~cTd;?A1Xd&RItEV;$=fCQ)&dl#^)pCb"
        "LCemxeW$6{kbUmi2JOEj`IPp%q=ZMALm1Qz$h4Uzuq()aLyJ8Jr9Gt?<ff&lz>uR<llk#uXy~y<Eui@oCS$w;NzA78&Hx9Z%}eqU"
        "3*Xo*vy!~=97x?cA8+o{LgCuy0u|Db3<|52^YN(Ex&$}1)69ZT1M`n9y5J#)Yc+(rTr84rQGE^q3M`pKFA|P|bv{NLX$~4`0x;>{"
        "d$!qT?}bTea{Zgp*-RI^G7-gOwiPwa=26$Te}&hGGs6#fnZDwfV0*cYL^%}X33o@R9D_b2h+;>FIDP&BL5X#Fp7G(}`j-{rH45SC"
        "xnzG1Trw=3x6mmX7>h@Sp<N9<(|#ibAuOOv$lM@6s!myJdMyeVE1@8tO3hj2v~hMZjUvgSB$LuaN|R|`=bw#>C|nyL;x?cGgIr=N"
        "$fo)3i%($c<v}kT@_0~zusEjm-QzFdB2&JvvDNCQ)B61G$U3=gjZ(ysAiij%nQ@>#o|ewul}^j{P8|}Ck_IRUsHB!wBBUGSg3Fj4"
        "XpGTac-e1GjF%04qXiTUw+x(k=(ZQLSY|j>K@?*vU&b>)P;&uiXw0R>#)f^5ODvc@u&aewq_@!2vKFjVaHYl3X1Uj|-z0-cG#?vX"
        "C!5|5ys0N6*>><Zm=t^`ISX!itM%!N9FEOX*Bs}1#h9=S$hGLN4VAjWQLe>hf`VIkJ>bFt#Dnqko=S+N8+S~)TZ~t=0LGhUsL^V`"
        "2kMzB(;n$*qL9iAvDO-8$}=w#Phyt%<I+`5Fe$AQ4?>(l;~ZEV2zBnU3&R~0Kb=%|#-+CqYLmi`RDlPEmPs?<>PNm|QNAPXlGb$|"
        "aZS25d@Ug~rB;tz-+PY*)eUHlvz?8%bn;ox-MYy#bn^rN-Q_S-$H|;CQQOlzE6oMx9fpqm`OmCK6UK9OA$N-B7>|DoPZ!%EPn9<0"
        "nF-^g@VBVI#okw!<Gk^Ysc89D#ifd?`<RR?Bh<R2eGl5?c@pu?{eC~QTF2S(!o0oQiz0VLPKVC@NykZwSn2V60(=yH44>xVDNFa}"
        "l-b-v`Hr!k@y}J;LpPl~V{~#}iBoNyx|e(UTVCmA%=sr4;d$Q}8KP&GwsOrJ8!O<hWMwDR3Y?8@h%{Xv#H{e#pbHxU3sSud-K=e#"
        "-Ya~a1;dH5dLeAG{h1J=gA_H`*oF5xXecmCK4a}GjB5SVro#en))sLkKfb&9HC<+p7qd<Gz;LQ*=2Cfz(XmBa+^I(={kGDGpO2fp"
        "q)Cq?R>4JhZZ+sOgDV?DsoetCj9~7=9%SN}5&WoEvPpGx<<RDRiOs6X18<Ck#u#4jElz8^$r*gxq@NYsH2TtfK1^n}N5@<(s;%t4"
        "Fx<ww6pc1idkWlKUmZljRn@+gfKH{Kz=^miwF64^{zC<G-+d>uX+O4dYsPhYO)!g!%q_68{gp00sJCrBe2ODHSWn^P|Knn(y{e^x"
        "Z<*3#M7=zc#`1GCkm5?xaK~m9o$R~!{}zCM(O5%oM&2>=%rylw5nZ5iZ^;EgM$U3rW8gK}Bpj<22Hfq;L?;BhqtUL+@Js&h;zjpj"
        "*y1$~@d0bU#hEouU=T_DeU^*-qWpG2awFR}<f4Fkg<qU(9ZeIJ0GKo}TgWGl-h1Sg=MtJmyXnNmQGknRQAir_^Rxc}P)h>@6aWAK"
        "2mk;8ApmTEmJ5ji004;s001Tc0047xV=rQ9W^ZF*Y-eR{ZDB8Ta$#w1X>MmPaB^>EX>4h3XD)DgZIVH2+&~P4?*{(|;Zp(|yU?2t"
        "g&dL|N|&-ta}CN^;~g?Hay4={+h1SJrkj!ypM-6__w-3!*WG>iI6@*UtIz~{l<9y%un7AQsfR7Y{qFiJP|=W32|@uCvT6b>fd6`h"
        "${Mxrx~^^7#tSES44xgvnNq^uarR+`kQNq&o9BfjB+B}IA@|<@#i92cr*p@xnVQj-bw@ZIZ$wzCQ`=1B&w(H4RrjKp@c5t61}uis"
        "vpy(4Q7$8dtQ%02nclp+gsV?51*c=%fIT+W?OOduI8d1PS_WPWN%&1NpwsXRHE=FNq3Ex^+z(E;6f3G@oCc$sAR4ypj(rVXMR#<X"
        ">`Z-QQ;-YAbSzTM0+I5pd*hr9cZ^0ySzsZ2Do|)aF~xX<O7BIIyr-JRR%jm+DM{(!?!fT*+YfNKpbHwv6m83y)s|S+Wb0FgM{8n^"
        "H1)4DzZ8O@(7Ow`s0Y+-pxfffH1R~S@}pT!K<d2+{aiym5pUUz)r#@&zCxb+?y_zE08mQ<1QY-O00;m803iVA%C(=s2mk;t9RL6;"
        "0001Ua$_%IX=ZO@VQgn*ZEayMbaG*7ZfS03FLGsYZ*ysMZ*qBGb7pidaCxO!-EZSI5`SM{{|8q0BDGpy9JG7a1$+;8$O8A!V$ltX"
        "0zN}vWXdL@uOlhPZjk@IGyD=MTh4B)heRSd!<pfazZud=lKjH@PIA?W6Jf``6N;$rk+uAHs(9BD+HOfhRk>%|cYRm$@<cv=yeCyB"
        "2(4=(`H_@eTk+i>w18YBN%Hp1+c%Zy8q!m>ulWPvjn@1LK7HQ<uey@fJJvLGVRX20cfWnQ`{lns-Ti!DeExLzpU*QAO3+eCCv<Pj"
        "W?S%xa&dR0O<%K1qFbuyqN}Q!x2%wravIXNZwNfS00n{-Y|9mIcZFbOC$=&-y|S;ln(I^ap=Lz|Jxuh1w+cG4Srfz<dTW7TbX&-7"
        "5G5-__atXUqcyU^bqB%+CV0i!wy-0ZxTFIj$6K)SQWPq1FKE6hZV_Q69S~Sy+KZ;!vN|VQR?(qWg%c}!(N(-=60z|14Rkxqf!97J"
        "Hnw(AAR0Nuo|QRaZ3%h>>a{Q5`o^N>4Ogr0Z+?95`)%7lmp|UZYJ4#Ic!eq#JEqb^bTvyd^394Q+Qh_;(eqL+8Tpmg1G^KV6KTQ@"
        "@Sy|mR=+^Bv|ZpJu2GU1EeKS0RAhw{t`qy2Y)r4Sn{OqIPy;|VwQURYKeZA$p<4h4gn-l9EU#SMi-BbsdV?KSLJ(Gi(P`#g4Lbwp"
        "U>%wsElbBXe1H_xAX?J3^@&h2v>H-m%d3i1Lk(*O=-3gpXN0#>iJ{au9HeV?c39IF|7(+YhVM);0%;cze5~Q<#U*S1oOxgbm;p0e"
        "qm1H)NlAAMgTH<@NccdKK{Ao##4J{;pvg@BvhfsA0rRaK#1^#g>r+wgDS&_h?@+~pn7T=FkMq?>3eK4&-LaM#@+{0OdQ}n9noZQ6"
        "w~yeD5iv?_fWX=MtBrsN1fEhUInWmP7`dxgM^OR$qv(<Tlg-N}AC*n~7&|UIG#z!(wLJ38ggDw4&QH59%!rHILhH+dC|%Qo{R;s&"
        ";eDFDu8vn;<N5=sS(}cgW*cOY@%t)~*kLxx58cpisW>^>FwJ1IpKLM}`LOW@0r`XhgXBDZt#7nZx7do~tYY{wZ|iqUJz#nB{;IhF"
        "Ge_aEWS@-TK|=#x%gDcCdd9hNjPoob@5mAcMST6Gwip{a$a_`YJvl3cVTj8$LF&1OwvcG~HreDM5$8elavP;=%vG*F2#v7B?<xqW"
        "b9}h~Da)>?%g9AZp`Nm>-{x^52ojj-IL~5EA5LnUZ*OjH{P@Gz=&WFDyITGuAM>9pTp-=oiV55tGj&fuA=C0CMMq)Fn5I>Nl07XS"
        "D`XQvpR66r8+@YOZ7~ZaWGZYD*0UAr)MGAQ0m}#R)f7}5mWeJr1<<;4AiIi_`#qOL3@z{x2>gHG0{Bl%D_)2K8q@G2WRaTfD3ml-"
        "{lWq2Z{(w3>~{u92hv;3cN_=|Qpp|*A6kk|N?#D`vPY5!2F5b76W!2*VqG|5U>8o~zo}$cW=XQA2L_xAbr$-EiA&Okkpq`@8dE~_"
        "^&lM{EH0(hrXdB3`22z}H8`jUusaM%0nJPRz*rb9tbknmLYoYD*MWfOYIVV2KgIf7W&_jC)Slzaxi0?G$sTZ=8f@9wS%Al!YY#SX"
        "mQGd^digHiGw}Vx^46w%bCk*5OEJQK4J3fk(<uW{);$F{3u2+fX%ZP6UcyO<L*0ip$Jqg9GCpL3*@}1iU<G`+$4@=eZ&Bcfxk^yi"
        "jc2W<c>Eb~{-x?{w6mG<F|$QqGZHc7#fv#_a*T=#s<XJsLIBGq)^0H=1OwQyWKe95WCCszhA)$~1WuGt`>x>dCsIA&@-z-<Ya@W0"
        "X40d1(DdfyCIi?mBd#7L0{+KS&H-n?bS#ai+w#3m1^6eA5=ii#2PwqAMeOYlUhq;h6zXp#)fE=d#}6-bjR^eN4qigdfhSid@$#>+"
        "&E!xNEWAECY540+kVT%CP^Bqpc~&oit1vW!jO?M;1F0Y#mnmdlW=atmh&fS9JX*s9Z0FPy>T*cFys5sLa5QXYNe7>xxViMW#h4%A"
        "F!8F6uON>=?;L@gpOs(b4t%m#Ddme<1D{u!10MspfHX|QELY&;G3fx|0CHY@`JvbAs5>~soeW?69r=b<3M&v4wgn{CHLg|)_U-ei"
        "H34h`f8n7sK%gyc3Ou}Hffs~6F`Q<P7a#__-?hAigGZo1msiEUtGC^tB5_+$8v6<pH&A2QHXP}Q*pzc;5)+=jA_)fu7{3;i)5TfU"
        "8Xlz+bAp*pYF?sn0BJ*q=I)?5+7r+}w?~p_8^FN7vMrL4T1`0s`ktLB#t2W}xu4oh>N{RSg)U%P)8snJ-pY~naS_B-NA-wN^(clK"
        "aWmPxF!$k$M3<3hWSc$m^a75VEvi6;@ifopTldjiqdacTyKwqPogE^e<Zi0*4WN#vfUea|E$!vLQ-$2q+aKOzlZJmXA&!^x@-(hy"
        "$NqC9VlcuhY=qp|z4jz!tj}Fsr%}Xna$H15?4v6Pitw;~T-fUneJ$Wui7VE~nK<gcXer}y;k}}D_9f=2brQO}(TsfNup1m-Foej<"
        "BQWJ&{djYLSL@+5O6B5cF3Q8Q&Y$J&8<1N+xE<(sR(yrf&|@JL>m{DJ-L*pBm3wHgpR_E8W~lY$-0J(>`67n~ycJ!uvUR^p2Ni92"
        "eToY(Pb7H%8r<swgJ?fIg}~Vx)lF-wM4p<4DZ$HRl+oOyI4c}T+$2xB^wp9H9VO$k%>E9sn*XUG&DO@3jVt6X{Z^6MUo1?UulbrD"
        "0`|Zm8=WROjf!#)04xCv`Z~3Rfd@8O<Y``G#~u=9V{Tdl_ggsghvSUFn6mUs#N|}c$tImabF{yGR9sFqFY&gp(4PDb6y78UXq_T9"
        "MlC+N5;yP-Pl!=egZ2@`{w=&XHgGoo!fWRc*Wt(1%+dQ*<ok3V1n=X$^dC@50|XQR000O8001EX+W}19Bmw{c9|QmZBme*ab8=%Z"
        "Vrgb?V_|G(Wo>O?FLZKYX>Mt5XD@VeVQFq<axQRrg;Pyy+dvS#8~h&zof1%G)20_eN=ssw9AZLZ=uwP!XYHl?q3%aWLVtW`W!Wyp"
        "p*l#b-M2II-n^-*YCZqhg4BUTf~VygWYEtPpf@OJ=Z&P$z!MQjFXC+d`0Z|WxBA{#*;G|EolGYxcncB}3WG6H1Z{gC(rIgi^q6)="
        "Zw|)~{Cm8<(aH-nTe21#;hoak18(usQ_z-Ndg@7>!$*NiaiNDBOTv@`t$cbUl&^~rd?=Zwlds2`E;!5^x$AYvbUB?2dxibYkh`5u"
        "fFDB`oOatTV4@b3@tEK@th^(*9zJ`V)gt(AgGubVsNWc>-C5u^pw!HyLm{%w29x%#i;4R9?j4IS|3UQJIvqJu3o(VW2vUiI6dK#t"
        "cTtH@>?m}Ber2|UY+>WQVaO6qq`iWY6mT5vy1iq%sM5v$dgS)B|Fo+fT+mr`T2V~x(DfSj9d1v8*Wh#Nb1G5W(HfWV;mc6n*bTPZ"
        "GMC1-%-Ab!r~o3FV%(u7-0)tH0WBp8u^itl%A0A4pS@@OnnI#PyPbcU-OjH7orL@O)$D%$8ElRTHUy#Ah8M~Q0^cWXxv61Rr~rFF"
        "vLzZ=?|39gVK!+LnX3jzWp)f^$2lY5LDU9T*%;WpvO0fNI7$tcFfubaGAjv=r1uNv7wN3ENBIXd&vI2m^`pcr<<{WXW6zwxOENV)"
        "ESBr~L}_4u5{Ju)*-|s6<hl3*P)h>@6aWAK2mk;8ApjsPk_V*@007%6000L7002x$Lq$$gMcrCkbKAxhe$Pz)!wzTKk!B#;ag(%3"
        "o@7go**cVml<Udk5?GL116X7i5XJubp3B}qQnEXJtH+aA5ccfZbNSA9&O(SU{JHLBxl`i3E>-8$x8FST>OYlr+H~U8{AD5jF1u0M"
        "gLw7w<@Zmo+6_bh^NSbz{eCWKqq(u$7Y%RbUVQrvExo^8{k9g@%Nz0LZh5n~U)(L%;_cl^d|claVs-mrb$9de4L&aDt((RAezkc0"
        "5ubPw-_6C1s&%Ia(9q5OQ}{FUCz}bklT9O9B|9;|i3ep{CuCQN(sUKC6Sc8obgB?m_108liI0mQ3tz6Z8?4@p_zNMOsBl+RiOoT*"
        "RmoDmgAJ@1w>$BZFg5(3;XPB1t?GtT)r>vStu+0?>g{e2X5T3*Kr2|Phl7ygurpTwn|2I>&Mg^sauBfZ)=F5}ZRsU%i-`_uD;sf3"
        "(ogglJ3J*FK?zxsoS+EU0^SXB47}l$)XMShVAsK#rV!Gq@Iyn27kDCkI(8K-D^1&)E{O7PioG6oEQq(9i?_y-CjDr81M!oL8(klL"
        "HS=O-bQ&iv^d+w~du0o7qXloF8oJ|u7h*7?l;C}QFNopakrk|HWhb`^y@)%zvD|sp3bEfQvH*kuZ7fOn#5#M8fdyhNH0VNJay#9l"
        "s9M)x(q7pTWncX8@=uqvp8*rG^`oS5a0A#LJps<L$_0@ibE7&iM5)1{6Y-g*x$A#7<4jz@YW$a-UFP10zi5~TU5zNsiabt&1oa3S"
        "Ylpgk?yYu?!k<EpQ-Zv6ChFF(VF@`2`PLo>U2j#btOWt|+nNmj88>N7rQv*%vMhR9cV#n@K_DwdX9m&eR-*#oP-p64kCEnRM{rIB"
        "P6)|Jl7bw5qX>{t>+NXi1#nNJCYW-!`4>dRsrs@z@H6ns*igFE*0k`;awj{`Jmw2TNayfX8G?(RH2%k0NWnHDdByaQLCDcrA&q*C"
        "31&z=f7UGo1)KvOPfk4nH8}1AgS$gPoXoANv=qakpPv1bv7b+XVh;~VYl0_?yOa{T3r9%_!3OkaYh?xK@*s61H%-8V46X{qS&YU~"
        "`mmA-D+20)E#U14du%UwTa$6+Fktl|GX;Hvj0-q}dX#Msd&5e=eTaHqiLYMwy@I_TA-fv0zsyZ>qpW@aYd$D}R&lfASio)08^T*4"
        "h+;zo<u}ry7duMM3OB)IfIwmh!@bCDm>T<?E_XRXfY%175Rj}^J!tYN#x&T+=Z#QcXJf-pAjA7R&#oX1>!5NFG~@&c+nI)P1lH-T"
        "?qHi!-=89Ngs*x6357T|p|>DLws#@P_Z7veR_c_ns+X2R25mzpX_ZyY0g|HoOvc+l*kHhPvQ?MC`5Ms4*0Q7mQshN6TJ}UOw2(4&"
        "?)^82^PUNu_55){#2n7IjArx+7^-bl1*J_~L-AaB#u7ytwh%3Y7oIvmk!8q$CBeYnO+<7zZU9a^o&<VAaYo9Mf<7;3CxZAB@)5mF"
        "_3+tJk`o$)YuX<}f1|*lHCW>rlJj+pK+NK)W<eNpf(X~JN;Qy<)&NKs=!%VOD4_Nh7j{HZ#?HGyU_wkRuadFQR)b54Lr!qTvt=s5"
        "S-uPWO<Dm;>IPRgKtMrUR?i}1atAl4)=i)tit?zixRg{v{VN;=SQnYoL{gSpzQ|!^5)7Hi(RRRz%h3@Jpe<X%P|rC(5e8;eOFafd"
        "OphA^6HeoLU5;jSkjkyJpAiggB2kf>DW|s`)dYx1^b}e5Y)B${&6Z$vDe}CX&(0<E5hIUB59#=2W|W%|F|$3|SL{Haje@`edZ`HC"
        "LGOH{EF+!zV}w9zaO2W|)w!G_KhAP7K+~(a_!S8l?))YiA#gCU9=Y`Tusz!a<w=)=wSpQdGAjwh6wsRpA2A0Yl5kQWgS{HSJTVdh"
        "4x4JPk&kqyyP^<qaBTeX3b>KoV$Wp`vKbCnwN>zw1`7CKO2qfmYVSKp+%j}(umm!$$Cy3=zYOO4aRZCNmJqakBO&VJ6HuG$u%joQ"
        "dgMLdq&0{LP4xO?&okAYz$W<VFPWb{NW|j*4<}u~I@J%DaL}m?fk=UhjytJK(erULCj!9-X*>C#hz&-Ss24YNjf4=YgK7Yk`M&@%"
        "#t!VChzy=qd-_Po6Ay=GU{{5GW#2d0teOs-Po_pZ^U9Tt)L?XeH$NYklBDD&j%eCJHajP+rev)x08(g#l#X>VM}iA?34OHbl&_Y6"
        "i9lLn8$j!h7sNAhhwN)HoPf#TR6W1P9{myO;9M+f41?4{IY2Lr@93IAZ&?YsmH2A{KHvLaq*|7kxHZmQk=5`dr5TZB@y`$dQZ#b!"
        "Mm^wZ8@1)?09Fg?XPn^(DxU*ADi*A|^Syc|s7x+8gp&oYw8Uva4)e8XNJd%}T3Mepp*2bQ=F4GVUR*UWU(ts#W~2+Wu7al_3ZscZ"
        "9JZKMfQIkqVx{sy)0}p1<sqTnF)#r(>cD@d5cLcjB+nqfgxy8}J_;ygJ@9{%D)zMR<{JM5x)rH^BCBLU)+)vBu1y1-6cc~|XFsPp"
        "d?7FSxFbZ}7Ij7yxkmv9Yd9wYS<aJFV~KxGPAjQ$95)}oaG8zUZnAB84wR@j_Fp&z;2Dv{aDtXYcNjC=dAeK!1il#IC=}cOE!jk|"
        "pySQUY;&syNVy^q?Q9M}9R(<Le8PP1xMvEE0&^}^&%#H0fpA$VWLQN`IVfU>lz{$#Jjytu(FxY#pTzPQR)Y{yuQEh{p#<Sc(Wsn*"
        "cF4*aALg{0ADgbaL^O-B;M+#@`fPc3zj$*ygM@q>$nKblej8-z`Br(pWx#MIbx!O{Zp&qb)>VQpWkoG&ia>SNxQJB}C&{@Sk9LF?"
        "d=@%I@io@X<(xP4S?f~RfsjT?hmF|eO2Ds42@DJhw*5IMDub5E;K@qU_;Sx`@k<W$lW?14`gEctbe+%#Yw9*t$kPpsEzX!)240xW"
        "&3s33#!~fh+7Qct_K{r%5^Z&b=R8CYbvW~bjvLvBlF-85@3?_OP&;Gm%<aT5xEGI;L+CtG?~6R<NFkqTgkp!u9Y8E8va0ak78|8J"
        "G;=XQdvB$$Ns5A9;J_P`V^J%HQ<ADuT{X6W<W3?dAPqN<!Ko)OM^+BA5HQ8r#G=_3G+WFS?06E0Y@Da>NX{B6^&-S_=~94M!qFrx"
        "_XtYyhso8(DW67?J7L9VNPjjiJabLOa<c0wQ=ieN$a1KrUi0uo2a^x{VqTFz+&iBTCKXPu$R=ev@^l<k6Zgh|nof9PgLBk99Qox3"
        ">XQ7O1NXX#GrGC>*g=(b<U92UdzBg+Xp)&-$D~Qg;h1!0)3IzU_T(rmYY%YG<Eao6!%aTj{;#?^&&Wv8EOJ-|Q{5`weX@Y18E{p+"
        "E~3h_G2G8#8gHpJ#QH+Yxe>~QQ<dVYC(Mc5Gkz;3AUt{nQ^sCu3;iG^!NF$>HFN4wl{xei9!H~CwUw5yvW{Dpc;)aTAY@=E4$(5>"
        "-jyLh9hjzPcTi}4f04%=C*GLI7H4sh_uwE+*$2Ei@IQiDKA!pI5Wqp{7&58Nwd#*i`}-;^_zqlyC7)ai%)qpGQI7h6t$~@8a7gcv"
        ")U4nX_+<7Qax1vc*NHQ=A(-p0b8({`wOx2`T#HY@R>3%jm=RIi&4D{$>fW(E%YcF0No`M>tQE;|9t~X5_5#(#(ZflPo4<~O(TSHY"
        "aZ(1wV0OJ0i}g&rzFsfZ(Zrt?_wVjL-iuGytJU@LesQ}NcdPsxfxEZjdilHf$6|Q{oJjK>^dk;lU3M5vSW{&q*pyE+gOvfh4$vl("
        "xu})0XOI^xdcU}Te_Md@msgAB+tp(E>+Ns1%lkt7cDs7>4)nZ!y?DR4|D9s#?c#oUyXLPMu7i*d*DG+`$M@GO@!{j@!`=Fpt1;hG"
        "Hh6;pN9bW&O}86#3&c(3QQUw7t?8{sK2L|PA<pm}io^uO*({c)n9c!l!V^ar*N!0HnNr72o6+81^U|m{zvn&ehC>woZ4OU@VR6}e"
        "EjPNMOZ)|vW&tFr8%PzF0*@M+qk@{yt><H#aCZigG~}a)PHh_vEUi?RMZ9z`CX?bcNBUyyUoe-zn}bGgi20FvTbyg83zD$ufS*G+"
        "x}`s#D~zd=x`G3q=($GI)_$-+E|#s_PHzfudH6aZeW&1hg?Ie<1vq3EuwcHjK$^o7W4r|PQUh?~7!NeZfw<+Hd#v)QmgBYcaidHo"
        "9V6<D{6u%&r5V^x<_;InUd#u*@Z^o*(A^qS?R7JmNPdQzZTeo~>=9YRh`QHOH>2f3C7XKe5&@zDdG6a3yllWY&TYb5D+d9D0f~hE"
        "c(xaW#DTr69yDFg*8Y1E$OdnS@TrBDAF|=6xwtN|SfD8a9^>ZMsgP$0@@a=$f114~mz2+5@dd(K?u_C2DveO5SN=551%g*ALKuKQ"
        "(ouG$;)C=&VD*T2pa@iLho8G-Gd4DE6Vw-G)A+F#ahn$iL`akQUI&hZx#Qb19bq*#CGX50`zP)~qtVE!ndEdt`p%|n@{5;9(fth!"
        "O^E$tMAw9^q%pCfbYYVLeHuGu@!%)U*lOvTQ5*B0)1T~9C;L>Yh8~BP0FSQD4Cke737P^GkCx1`d$cy)(D}hMAT>1g*y!?nwm6+i"
        "Zw{V~WM?{{6_Q~i`QK+@neof0O^D^&<qcNdb6@QFH`1NctCz2Sgj}$r`8>#t__KJeYg5YRSJk#M*VOS!9|Tp#k*8lL7WpTL;sRfY"
        "pZ8t<LNWsTM4vJ!`Qd8dYr6Bz2L*bZ52OMnQp0C9G&PZv;Y|QVIiwEpOW!%Z2b#8A-(G>*e$m&Ev8O7F|D=iJWaE|74`*<&oWJlS"
        "*ZLocR$y3cy|q#kCBlU<l`UvP;dI@01b_s*hg9#5KOYPIa-wGL^k}fve*jQR0Rj{N6aWAK2mk;8Apl*wQ8u^&001@z001Na00000"
        "00000004ji00000V{dL|X=igUWMOn+b7gd2a%F5~VRL0JWq4t2aBO8RYIARHP)h*<6aW+e000O8001EXq%`6q4*>uGya50JApigX"
        "0000000000fC2gf003ieZf0p`b1!6JbYXL4bYF61Y-M3{WiD)QV{0yIb8l`?O928D02BZK00;m803iVBu1g=*0RRAq4FCWb00000"
        "000000001h0Z0S@0Ap`%W@%@0FJ*XeWpZh4Wo~qHE^2dcZcs}B0u%rg000080000X0GrNDBV7al0F4U(0384T00000000000Du8h"
        "1^@tKZ*FF3XLB!dWpHnEX>@OLd0%sMb!2%iYIARHP)h*<6aW+e000O8001EXT`4=JW&r>I?E(M*8UO$Q0000000000fC20Z003ie"
        "Zf0p`b1!sqVQFqIWq4t2aBO8RYIARHP)h*<6aW+e000O8001EX=Zh95djS9di2?usDF6Tf0000000000fB}jO003opVQp}1Wpgib"
        "WpHnEX>@OLd0%sMb!2%jWp-g~Uvh7Eb1rIgZ*FW*O928D02BZK00;m803iV7V;l{_0000R0RR9a00000000000001h0Za}60A+Y#"
        "ZE$R5b1!mbaBp*IbZ>HbUvqSIWO*-AML|SOMJ{b*P)h*<6aW+e000O8001EXdo9;C-~a#siva)tX#fBK0000000000fB{br003op"
        "VQp}1WpgibWpHnEX>@OLd0%sMb!2%jb8ca9b7*gLb1yMsW@KSzHa0jkHaRjdV>M=GIAu07WivT4Gc`3ZVlg-{H8nIjIAt<8Wj10n"
        "Wi~f9G+{DfG&DD3F*iA5H8d`2b8l`?O928D02BZK00;m803iUpmUo)o0001s0RRAL00000000000001h0k#kT0A+Y#ZE$R5b1!mb"
        "aBp*IbZ>HbUvqSIWO*-hZeeh9Xm50LFEM3eWHDqpV`MQhGh;C`Gc{#1VmUT7H8W*qG&L|WVl-qhGc;ymWnwsCI5{(AW-&HlFlISn"
        "W@R@uFgGz{H7;s%Z*EXa0Rj{N6aWAK2mk;8ApqyD`R3pN004^t003zK0000000000004ji8xjBjWq4t2aBO9BFLGsYZ*ysMZ*qBG"
        "b98lNc`tKrVQ_P3Z*+4nGh{R|VlrenF)(FiW;HWoV>n@9H8N#6WHV$mH8(V7HDWU`G-YHlVKgvhGdMFcH!)^qV`VvJF*r9kH8MFi"
        "E^2dcZcs}B0u%rg000080000X04^)lo#6lg0E+<t0BHaK00000000000Du946952ZcwudDY-MvVa%FIDb7^#Ma(Q2KbaiBTFLQ2T"
        "aC2yHbaO8>G+|_9GchtXWnwfjGB!A7F=I3`Vm2^2WHvA{WH4nkH92NEI59FdVm4%9Vlp@|G%z?gF=An2VKF&oGBh$SYIARHP)h*<"
        "6aW+e000O8001EX7KqqE-v9sriva)tX#fBK0000000000fC1_h003opVQp}1WpgibWpHnEX>@OLd0%sMb!2%jb8ca9b7*gLb1yYG"
        "HZWsiVK6l@Ic7CuH8C_XGc{&rW;J9mWHK~nHDWkpF*h_bI5%N9IW{?DFlAw5G&yEsIb}0sIWc57G&wG6b8l`?O928D02BZK00;m8"
        "03iU(2MA%{0001r0RRAL00000000000001h0Z<kI0A+Y#ZE$R5b1!mbaBp*IbZ>HbUvqSIWO*-hZeeh9Xm50LFEuz~WMwpAVP<79"
        "GC43|GGt<8VlZQ5G&nIZHZo>8Gi5S3Vq-TpIWsk8V`eckGB##6G%+zZVlgu|V`XDFIWB5*Z*EXa0Rj{N6aWAK2mk;8ApkQw?GfMr"
        "004^t003zK0000000000004jiw-*2aWq4t2aBO9BFLGsYZ*ysMZ*qBGb98lNc`tKrVQ_P3Z*+4nHa0OZHDx$uWiw+rGBP<fVly!^"
        "H(_NsFfukVGdE>1W;rusGB;%~FgRp0GGsM5I5=ftW@I;FWi(?jH!x&3E^2dcZcs}B0u%rg000080000X0CkQ;P~ZRn0E+<t0BHaK"
        "00000000000Du7=82|ufcwudDY-MvVa%FIDb7^#Ma(Q2KbaiBTFLQ2TaC2yHbaO9aIW{z8VL35mV_`L8H8VIdIWc8nG&eUiWMX4s"
        "W;ir5Ibt_qF*Y<cF)(5`F*#*nV=!bfG&MG2F*jvoVPa-3YIARHP)h*<6aW+e000O8001EX4ayOb-~a#siva)tX#fBK0000000000"
        "fB}UX003opVQp}1WpgibWpHnEX>@OLd0%sMb!2%jb8ca9b7*gLb1!9NVmUc6H)1t4VPs@uG%_$TVq`fnV>dE0I50UlHZ?eAWnwaB"
        "VK_B7Fkxk5V>L50W;bMFVKgx|VmCN7WMnRCb8l`?O928D02BZK00;m803iTY&)elg0ssIQ4*&ov00000000000001h0qh$90A+Y#"
        "ZE$R5b1!mbaBp*IbZ>HbUvqSIWO*-ia$#w1Uvh7Eb1rIgZ*FW*O928D02BZK00;m803iSbyiLPq0001N0000S00000000000001h"
        "0e2n%0CRF<FJftCZ)0I>XJu_|VJ}}_X>MtBUtcb8c~DCM0u%rg000080000X0P`loJ(C3h0E`j<02}}S00000000000Du7(9{>Pz"
        "a$_%IX=ZO@VQgn*ZEayMUteuuX>MO%E^v8JO928D02BZK00;m803iTbT`SE`000130000Y00000000000001h0puhA0CRF<FJftC"
        "Z)0I>XJu_|VJ~54Wo~pYUtei%X>?y-E^v8JO928D02BZK00;m803iTk@JS?d0RRA*0ssIW00000000000001h0dpk)0CRF<FJftC"
        "Z)0I>XJu_|VJ~54Wo~pYY;SLHE^v8JO928D02BZK00;m803iScl}$BB0000=0000Z00000000000001h0U{>=0CRF<FJftCZ)0I>"
        "XJu_|VJ~5Gb7gdMFJE72ZfSI1UoLQYP)h*<6aW+e000O8001EXmZN1%;{pHxVF~~MCjbBd0000000000fB~x~0047xV=rQ9W^ZF*"
        "Y-eR{ZDB8Ab8}^Mb1!FQZEayRUu804E^2dcZcs}B0u%rg000080000X0JyAjh~okP04ECo04D$d00000000000DuA0DgXd;a$_%I"
        "X=ZO@VQgn*ZEayMVRLh3baO9fWo>O?G+$*jVlHZPZ*EXa0Rj{N6aWAK2mk;8ApnEz8iXAK001uv001ul0000000000004ji`z-(f"
        "b8=%ZVrgb?V_|G(Wo>O?FJW_YWpr~dbZ={AZfSaDa$jv>ZfRy^b963hb8l`?O928D02BZK00;m803iU%U1_O~0RR950{{Ra00000"
        "000000001h0c9}&0CRF<FJftCZ)0I>XJu_|VJ~B7Wn*h_Z)t9HX>MmOaCuNm0Rj{N6aWAK2mk;8ApnQMrm1`d000dR000^Q00000"
        "00000004jiGBW@Gb8=%ZVrgb?V_|G(Wo>O?FJo_RW@%?GaCuNm0Rj{N6aWAK2mk;8Apj!h{%0=(0046d0012T0000000000004ji"
        "<~RTVb8=%ZVrgb?V_|G(Wo>O?FJo_RbaG*1baO6nc~DCM0u%rg000080000X096G%QiA{h0I~o803iSX00000000000Du8dJOBW5"
        "a$_%IX=ZO@VQgn*ZEayMWMOn+FJE72ZfSI1UoLQYP)h*<6aW+e000O8001EXXTKfly8-|JsRaN4ApigX0000000000fB_Oc0047x"
        "V=rQ9W^ZF*Y-eR{ZDB8DVRT_HVPs)&bY*gLE^v8JO928D02BZK00;m803iUT(M`e70RRAL0{{RV00000000000001h0SrI@0CRF<"
        "FJftCZ)0I>XJu_|VJ~E1bYU-Hb963nc~DCM0u%rg000080000X0OQt%K@9@{0LKXc03iSX00000000000Du7&LI41Ba$_%IX=ZO@"
        "VQgn*ZEayMWMOn+FJo_PY+-b7axQRrP)h*<6aW+e000O8001EXK1jz6$OZra5fT6ZAOHXW0000000000fB|Af0047xV=rQ9W^ZF*"
        "Y-eR{ZDB8DVRT_HV{dMBWq5QhaCuNm0Rj{N6aWAK2mk;8ApoMNbb#_3006FG001EX0000000000004jiXiNYAb8=%ZVrgb?V_|G("
        "Wo>O?FJxhKVJ~BKa$$67Z*DGdc~DCM0u%rg000080000X0Dkycq&5fu0KXgn03-ka00000000000Du9RXaE3na$_%IX=ZO@VQgn*"
        "ZEayMWMOn+FJx(9XKrtEbZKLAE^v8JO928D02BZK00;m803iSfeN`B4761UiNB{sL00000000000001h0TFQk0CRF<FJftCZ)0I>"
        "XJu_|VJ~E1bYU-Mb!=>3aBpvHE^v8JO928D02BZK00;m803iT8?!XS#4gdg4G5`Q100000000000001h0l$X;0CRF<FJftCZ)0I>"
        "XJu_|VJ~E1bYU-VW@c<@Ze?F{W^^uac~DCM0u%rg000080000X0KrM{K}!b!0QM6A03ZMW00000000000DuA4mH+^Aa$_%IX=ZO@"
        "VQgn*ZEayMWMOn+FK}UWV`yb_E^v8JO928D02BZK00;m803iU23ODE50{{R!3jhEo00000000000001h0brg00CRF<FJftCZ)0I>"
        "XJu_|VJ~E1bYU-Wd30!RZeMYAVQgu1c`k5yP)h*<6aW+e000O8001EX&w;`9njQcE!gK%tApigX0000000000fB}G_0047xV=rQ9"
        "W^ZF*Y-eR{ZDB8DVRT_Ha%F5~VRL13E^v8JO928D02BZK00;m803iU;6&@}L6#xLxO8@{Q00000000000001h0a?KS0CRF<FJftC"
        "Z)0I>XJu_|VJ~E1bYU-YWo>0-X<>9}Z*DGdc~DCM0u%rg000080000X0Di6OkiiWA0DCO}03!eZ00000000000Du9c)&Kx=a$_%I"
        "X=ZO@VQgn*ZEayMWMOn+FLGsYZ*ysMZ*qAqaCuNm0Rj{N6aWAK2mk;8Apk@3Rl02)006{j001fg0000000000004jirsMzsb8=%Z"
        "Vrgb?V_|G(Wo>O?FJxhKVJ~uJaBp*IbZ>HbUt@1_aCLJoaCuNm0Rj{N6aWAK2mk;8Aprjuz0CIr005>O001EX0000000000004ji"
        "TLA$8b8=%ZVrgb?V_|G(Wo>O?FJxhKVJ~xcWpiz5bZ9Pcc~DCM0u%rg000080000X0MBF+v$Y2R0M!)$03`qb00000000000Du9J"
        "3IPCfa$_%IX=ZO@VQgn*ZEayMWMOn+FLZBfWo~JDVRUJ4ZZ2?nP)h*<6aW+e000O8001EXr3N>f=>Px#+W`OoCjbBd0000000000"
        "fB}gT0RVGyV=rQ9W^ZF*Y-eR{ZDB8Ec42IFVRUJ4ZZBV7X>MtBUtcb8c~DCM0u%rg000080000X09?f29bpFm0Q46C04M+e00000"
        "000000Du9v69E8oa$_%IX=ZO@VQgn*ZEayMWp-g~bzyXAZ*DJTc42H^V{c?-E^v8JO928D02BZK00;m803iT3r1#Dm6aWC%M*sjM"
        "00000000000001h0bUyc0CRF<FJftCZ)0I>XJu_|VJ~HNVQh6_bZKvHFK2UYIBPC&c~DCM0u%rg000080000X0Q>}I39bPE0Qv#|"
        "044wc00000000000Du9yF985^a$_%IX=ZO@VQgn*ZEayMWp-g~bzyXAZ*DJWVRCL|b8{|mc~DCM0u%rg000080000X0OsoL4x|eJ"
        "0B|J$04x9i00000000000Du9kF#!N`a$_%IX=ZO@VQgn*ZEayMWp-g~bzyXAZ*DJcVRUJBWnXZ4bZBpGE^v8JO928D02BZK00;m8"
        "03iUj6Yi+j0RR9N1ONak00000000000001h0h&Dl0CRF<FJftCZ)0I>XJu_|VJ~HNVQh6_bZKvHFK}{YWNBk`X>V?GE^v8JO928D"
        "02BZK00;m803iSr<!oF)0000(0000c00000000000001h0l7Z`0CRF<FJftCZ)0I>XJu_|VJ~TJW@U0^ZewLHUtei%X>?y-E^v8J"
        "O928D02BZK00;m803iS>kw?nl2LJ%%8vp<(00000000000001h0X;wg0CRF<FJftCZ)0I>XJu_|VJ~TJW@U0^ZewLHXJu|>a$$67"
        "Z*DGdc~DCM0u%rg000080000X0CnPo8z}(*0OSGy03!eZ00000000000Du8wNC5zIa$_%IX=ZO@VQgn*ZEayMZEs{{Y;!MPUukY>"
        "bYEXCaCuNm0Rj{N6aWAK2mk;8Apkg%l-P3z002H0001EX0000000000004ji%}D_Ob8=%ZVrgb?V_|G(Wo>O?FKusRWo&aVVsdF@"
        "XJsyMc~DCM0u%rg000080000X010WM3cmpW07C-+03ZMW00000000000Du91Q2_vRa$_%IX=ZO@VQgn*ZEayMZEs{{Y;!MTVPj}z"
        "E^v8JO928D02BZK00;m803iU#MspTb1ONb~3jhEv00000000000001h0en*d0CRF<FJftCZ)0I>XJu_|VJ~fOWMyn~FJowBV{344"
        "X>N31baG*Cb7p07E^v8JO928D02BZK00;m803iVRC&n9a0{{Sn2><{u00000000000001h0UcNY0CRF<FJftCZ)0I>XJu_|VJ~fO"
        "WMyn~FJo_HWnX7<VQ^?)aB^>IWn*-2axQRrP)h*<6aW+e000O8001EX>gOZ-$N~TWB?kZiB>(^b0000000000fC1WD0RVGyV=rQ9"
        "W^ZF*Y-eR{ZDB8MZ)9a`b1!9WVr67xX>Mn8E^v8JO928D02BZK00;m803iUh1b)gA0{{R52mk;n00000000000001h0p(r+0CRF<"
        "FJftCZ)0I>XJu_|VJ~fOWMyn~FKKRRbZKF1X?kIFX>V>WaCuNm0Rj{N6aWAK2mk;8Appjtk{!ne004^?001ih0000000000004ji"
        "K4Jj?b8=%ZVrgb?V_|G(Wo>O?FKusRWo&aVZDn$2Wn^DrbaZ8IbZKvHE^v8JO928D02BZK00;m803iS#lcIF70ssJm1^@sc00000"
        "000000001h0ZwTF0CRF<FJftCZ)0I>XJu_|VJ~fOWMyn~FKusRWo#~Rc~DCM0u%rg000080000X0BuGL=>QJ^0HiYj03iSX00000"
        "000000Du8KYykjsa$_%IX=ZO@VQgn*ZEayMZEs{{Y;!MeVRUJBWiD`eP)h*<6aW+e000O8001EXt6AZ;e+K{nO&0(FB>(^b00000"
        "00000fB}4a0RVGyV=rQ9W^ZF*Y-eR{ZDB8MZ)9a`b1!gmWpr|3X>Mg?E^v8JO928D02BZK00;m803iSsM|gUh1pojf5C8xq00000"
        "000000001h0X~BP0CRF<FJftCZ)0I>XJu_|VJ~fOWMyn~FLGsOWpZV1V`VOIc~DCM0u%rg000080000X0D~B=k39we0J{(X03rYY"
        "00000000000Du7-i2(p}a$_%IX=ZO@VQgn*ZEayMZEs{{Y;!MiZ*_EOZf7oVc~DCM0u%rg000080000X0LB7QTiz4^07z8;03-ka"
        "00000000000Du9Qj{yL4a$_%IX=ZO@VQgn*ZEayMa%FIDb7^#Ma(Q2KbaiBTE^v8JO928D02BZK00;m803iUi`j2nk1polq5C8xi"
        "00000000000001h0l1_A0CRF<FJftCZ)0I>XJu_|VJ~xIVP<A;Y-BEQc~DCM0u%rg000080000X0NYRrB8~t60KEVJ03`qb00000"
        "000000Du9|ssR9Va$_%IX=ZO@VQgn*ZEayMbaG*7ZfS03FJE72ZfSI1UoLQYP)h*<6aW+e000O8001EXPWSxqmIeR-KNJ7}C;$Ke"
        "0000000000fB~Va0RVGyV=rQ9W^ZF*Y-eR{ZDB8Ta$#w1X>MmPV|8+2bY)~;b7pidaCuNm0Rj{N6aWAK2mk;8ApnJ~y5$}L0089y"
        "001cf0000000000004jieX;=nb8=%ZVrgb?V_|G(Wo>O?FLZKYX>Mt5XD?)Fb98BJY+-b1Z*DGdc~DCM0u%rg000080000X0Bmf7"
        "a~KW)0JARu04V?f00000000000DuAFvjG5ea$_%IX=ZO@VQgn*ZEayMbaG*7ZfS03FKuOQZ*qBGaB^>AWiD`eP)h*<6aW+e000O8"
        "001EX8e_bhSOEY4lmY+%CjbBd0000000000fB`|m0RVGyV=rQ9W^ZF*Y-eR{ZDB8Ta$#w1X>MmPZ(?d?V{~bDWpgfYc~DCM0u%rg"
        "000080000X04?Jm?XVC40GB@i04M+e00000000000DuAB!vO$ua$_%IX=ZO@VQgn*ZEayMbaG*7ZfS03FK=dMY-w&~Uvp-3E^v8J"
        "O928D02BZK00;m803iUhHXr5_2mk<P8~^|#00000000000001h0o2q10CRF<FJftCZ)0I>XJu_|VJ~!YVQFq@Zf7rWZ)|B}c`k5y"
        "P)h*<6aW+e000O8001EXY=D*vi2(ori2?usCIA2c0000000000fB`7o0RVGyV=rQ9W^ZF*Y-eR{ZDB8Ta$#w1X>MmPaB^>EX>4h3"
        "XD)DgP)h*<6aW+e000O8001EX=*qR9zz6^UFC72?D*ylh0000000000fC2U10RVGyV=rQ9W^ZF*Y-eR{ZDB8Ta$#w1X>MmPa%FID"
        "b7^#Ma(Q2KW^^uac~DCM0u%rg000080000X0NVjf-y{M603QSZ03-ka00000000000DuAh=K%n7a$_%IX=ZO@VQgn*ZEayMbaG*7"
        "ZfS03FLZKYX>Mh5E^v8JO928D02BZK00;m803iS%Es_VN4gdh#DgXcn00000000000001h0c7d{08B|kMNU&iP)h{{00000PXJE<"
        "mKy*7F8Kif000"
    )
    bundle_sha = "50af7336b4208a01325fa97edb27dd1c2ce5113d2291773fb3336b9d18297c1c"
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
    from bifocalgemma.models.native import build_native_model
    from bifocalgemma.models.pretrained import load_local_native_model
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
