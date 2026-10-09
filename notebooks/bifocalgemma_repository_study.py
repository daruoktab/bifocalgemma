# /// script
# requires-python = ">=3.13"
# dependencies = ["marimo==0.25.1", "torch==2.11.0", "transformers==5.18.0", "safetensors==0.8.0", "pyarrow==25.0.0", "pytest==8.4.2"]
# ///

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    mo.md(
        "# BifocalGemma · Repository study\nPrepare → SFT → generate → native score. Default actions disabled; fixture is synthetic. Eager generation has no temporal cache. Native Python execution is opt-in and is not a security sandbox."
    )
    return (mo,)


@app.cell
def _():
    # CONTROL CENTER: choose one action/run at a time on a shared GPU.
    SOURCE = "fixture"
    EXPERIMENT = "BifocalGemma-E2B-E2B"
    ROWS_FILE = ""
    PREPARED_DIR = "outputs/repository_study/prepared"
    RUN_DIR = "outputs/repository_study/training"
    PREDICTIONS_FILE = "outputs/repository_study/predictions.jsonl"
    SCORE_FILE = "outputs/repository_study/score.json"
    TOKENIZER_CACHE = "outputs/repository_tokenizers"
    MODEL_CACHE = "outputs/hf_cache"
    ADAPTER_CHECKPOINT = ""
    PYTHON_EVAL = ""
    RUN_PREPARE = False
    RUN_TRAINING = False
    RUN_GENERATE = False
    RUN_SCORE = False
    VALIDATE_GOLD = False
    ALLOW_CODE_EXECUTION = False
    FETCH_SOURCE_ROWS = False
    FETCH_TOKENIZERS = False
    ALLOW_GIT_NETWORK = False
    CONTEXT_LIMIT = 4096
    TRAIN_DECODER_LIMIT = 2048
    MAX_NEW_TOKENS = 512
    GENERATION_DECODER_LIMIT = 8192
    RAW_LIMIT = 1000
    TASK_LIMIT = 32
    PER_REPO_LIMIT = 8
    MAX_STEPS = 100
    EPOCHS = 1
    ACCUMULATION = 4
    HEADROOM_GIB = 16
    return (
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
        "P)h>@6aWAK2mk;8Apq&FOCQz&004*$000;O003ieZf0p`b1!9haAk67ZDnqBb1rIgZ*JvP!HU~35WO$#KLnq<)RITG)nlM6z4um%"
        "8A~ItK(>RdY)hB^du2Omv)iOg2n$Jy5i&EQr+Io0>FM@nu}B8>KrL2}<Okb#$%h4dH^qZT+u5$EJSn@Pm`JpZ|DhU8Nah-ChhUw<"
        "Ft0S|2AVVUeMR23)P*4?zKpTO;%S~@y2ExtTw2pBbobP@c=wSn$1!71Wz?x&laH0Zm&RxtW3IA#1ZeucarC~K0O0xZSnBD~ZfGT%"
        "OJnjHgh@+=Ac=U9WmpQS#*b?$G{f9H0~qX&snNfHdpE^59ac3Clx*`q|1i*WJcBzM4wA@-bI6&=1(lMCi~*PubzP*C)urM}$r1z>"
        "N0HbxS;c+#P{0tPjB)>JY<sM4$v32TbPATL?MZCGzV%*@+6TX;oyyl;HJ{rdXU&8j&{MUVx}Q>WX4~{@K7-{rPVyNX_8BmS$cnlE"
        "i6{lvhGjx(t`%XCNmif$&=U7c^b+{Zd<OsY8Bt0mPkEMeri>9O=t7h{=Nc%5QVJAk0QpfQ{x_fT(|vaN_t~}i?1e?n!f0^%!+H4^"
        "+O<8L=V7$_o#XxVd9TH~wR2Z4oWk{h?lOggHxupxuJ-d4Zk)c}-o*JAP)h>@6aWAK2mk;8ApknoanoP~004~(0012T003ieZf0p`"
        "b1!mbaBp*IbZ>HbUvqSIWO*)Xb8l{qR!fiEHVnQ8$bT^Q9Czf0{Af=_n-pn#YKry{6owwuj<UApNb=51g8cVVw%3^?UT;p4DDo32"
        "lArYP`Ligxxd-dC>2H3Fk{6R;80}yjpy0*=&a^l75)0FNIQz)?QC@zEB=am{vePaYN7_Zb@ezyE*tSv7Wu5Ms4|@Wv*Gh|Nbu*vg"
        "%xDETJmX_nqeLUT?c46~=HHj`D}4qV|81G><0K(}1Yc^%RFV}1hg8%oVQEHWnrA#Gq{?WX<5oo}N#+v}F#!E`nE`PO7@XHRNJOeR"
        "2VNJfRDvr`vy4|nr4-ka)`G7XJ#am`3Bm%0UK|5VRST&&<2mCAmzAQdpcUt7lFEb^S&`-{S5=klz{LNg0VyGQl}R9q5(QK;n$tQ@"
        "OM;#hqDU~DGGVGzE7D)-9Mm$#Z$|FG2dEUPgj7nQy+Q{Br$tq8CRMPnSyhpW7KNlM)GKF{kAHss%i%9f^UE`c0F$R&VoZdTDG4)~"
        "C_;E$38*RsKq)O{0V(Xj#IJkj?I<R<C=+Q838kC_!;(PBD#9|#^NKOSIcGv)yer<HzmqDVhhO3T<vnYsODdEqb0Vu!QN`+#7KkL3"
        "k~zohrFETVNtz_$i_A~AJl(kKv_HnfB|w#>V7w-j$~+DEqavlE!rM`GR+74qJjp1nK1VlL-3#!hkNb8wtiS;a@(S>>5~vjjNjaBA"
        "j*C3WnV_{~wMwC=RdExbKFxI>8meKfX6}pP_rSXMqqWfc!_oWUJmL4kNLbb{HZ-Mn!dUsk9GYdr;6Gdow`>M$PA%AAuuNmono+JB"
        "TSiRF*<_l;OfQlxT?_9=aNbQPIQ4nGr7Seo$)rlo60oBz(#nH$%fJlKGmXjg$BD9TX2DnTElY<i#%^J$12+zgV-l{#TUPJTc{^Fm"
        "vj@SJZfhiSQ)SGv2f=0<)&TqwSbJ+M)@bJrI{|R*h0*=M2IN!AmW+G=dpioDx3;%XPQbIJU{Ld9?0FS!`#FqrOz+-#5J!?^<Ux2<"
        "*1R?14rDXE7pVPqf!fLHBc=41Q$i0%xE$tTm2S2O;FVe!yKgYQ4Po8cOfCjBI()Pa^W;jtmEMk7ckEc(no~S^btU+$)QS*u7}~a5"
        "#k*C!->0F+c+14y7m{zfZn)gWA+e#tj1|Y`7EA};YH_6!j*SO{;>%$k*3HM+#R3yv-^1eeVOn49e{ZWjPAT76+B>xI1SvjFu}`=A"
        "(=4?|!#h2<LmRui`vm#zk;1yS$4PsP6IA}2DPim_IMd!ED@-pYcEers)${ju(Ov$Q&0tzBCT<!=j?)Hzmm@U1OKEe`y)-A++%LbX"
        ")u`3@zN1Z!&-c9J!@p7Yw~d6*h;~&;@wYXr0DdjT=|kFd?A-KlYP`9FK6q6nb-HNbXJ(v~Bq%nv0YkCY!Mr`SHrje*th;TM4$p5f"
        "Y=_0&&!L$fqDdU#y+wBH@zy>)fA&96O9KQH000080000X09`3Nre*;E0PO+*02%-Q0Ap`%W@%@0FLZKYX>KlMcwudDY-KKLb8l{S"
        "Qn79VF$~?2_y=@z1v^`Zwqik5#U7c&ISjcZO`Hl^_228<ag>wVF@Dd_&oBO(&lZa&GjPexdeKA&8V|u4Oe*1fp@dn*6f8ONZS&C6"
        "IRSZj3PV%G+Gm8QS!K&32SFcxMU0RvxQqj8C^I<+Ou<hPhmZzs;HvD4a|6<mPC@N~^+9F&oPuovL+Dw7)}<?hvdWAxPetJTM>W<("
        "^5EsJidHt_YNWkP5CaUV15Xw+_v=exH5Qh~9K5^c5DLe_O7f|^;gzqL3F1{uk1^CjUhUo<H`~qjW!afA%hD&(25`NIwelIL$cwq6"
        "`%O8hSd+PSBqVP`(pcrynCUcWJfkjsvqdy}(5ESW`~HvaA3*G$FipXjsDpG-HIa;cIKIT>R!@fbRfat8#$;k04(^`1r&b4pD;>Iy"
        "I11hJI$ro7`P^>rD3=yw#CV2N1JKr$Q*bWSq_5a~_C25d08mQ<1QY-O00;m803iV9ixwt(0RRAr0ssIh0001GcwudDY-MvVa%FID"
        "b7^#Ma(Q2KbaiBTFJ*RNY+rJ3cXKXkb8l{JZBpBA;~)@yU#b5va$j0GxQT7KNu)~YOTQwLg<+`CVxU~AM63Px0!^!IwLBo0bC`2x"
        "hCh|mK2XhYl)TzQCFChYhXmiLBI{}el|yT;kcSx+CnkoBP1NJa%(BRkDVt$xMc;0BoO0~?^^WpkViOg8FZkmyh{x@A$J!x3wa!cm"
        "mocdaTu_$;FY*M8Di#$D#x-9@+=C3@XA>0%hslP9FeCVAt&+$gnT-fmMGr>f3Zj?#wNR2<i>@h(G_X0Va47^3@|tqehfjiU>=~x3"
        "%TW_T{(+CWuInG1!vV-<)j^%N!{MlbqX5eJm~7k?SNHoA&FdEYStm<DG6>ZFhIP`&z@?zVP9i@;P{Zb^a{)>+)E;vQ&z7<oN!yeW"
        "eJ36-LrRq3VroPc(1+18Cv9o4aAvvcFv!Ik7rrA&pL&nik-P35o}Qrp4?@}kk&C@7H6b5Sze{KI*G%m>%tm+3ru+O&Z&A83a6g7q"
        "VN`!_etiH?O9KQH000080000X0OeyG4Z;8b02=`S03!eZ0A+Y#ZE$R5b1!mbaBp*IbZ>HbUvqSIWO*-AML|SOMJ{b*MUO#l#4rp4"
        "?*Z}$0zD@v`h&h<`vEm(*1|e=YRilD@2eb&-UN~wQE#}7Y>}oLD$ml?N*znH9tL)OS+8vKG*4fVjiBD6!K!H1d<S20oQVyR14<?$"
        "Q&>tj@F}T_v;H!0L4z3K*HHF7spn5Uc?~O^KOmLJ#qiC%#Y5@?ChgI1VV)XXy;gCQz;x9Wt@!B<wz1+?dUbzu(|b5cy;rwbDg$;$"
        "Sajw8N%S08q1Nzu$`-kNtnWy=F*mOB^b1f+0|XQR000O8001EXdo9;C-~a#siva)tX#fBKWq4t2aBO9BFLGsYZ*ysMZ*qBGb98lN"
        "c`tKrVQ_P3Z*+4nF=A$9VP-ZqI5ajnGB9H`W@b2LHZx^2IWjXfH85f^I50IeG&wkBGB{;6Vl!nnH#Rh3GGR0{H)An3Ib$_6E^2dc"
        "Zgr2{O2aS|$L|MwheNzkm|APe+P#XEktQcJGfhHrDvsg1+q4~`u&YDz|2V&M*=_(>#NgQhm^NQDMtly@pHR|c!FIFWiG)Y#hrv*`"
        "z3{HoRarMmD15DoMkwCNPUx!AYR|de*L5RHzNWRYn5NuYegRyTF$qCXpZj{W0b_a!=Or<V(mO&o?^CEjD0TD>?>~hd@!`<AY`TMV"
        "cGM#o2_(`76w80a=?n+6w{>jk{NCN#FNKw(73AYh1RebVD<_yC_()fz*0nAZUR(Y#t}K7~f1&bt-)^S)15ir?1QY-O00;m803iUp"
        "mUo)o0001s0RRAL0001GcwudDY-MvVa%FIDb7^#Ma(Q2KbaiBTFLQ2TaC2yHbaO8;WnyG8WI1DGF)}k_F*7qYWiw(qHZ?UfWo9%r"
        "Ffn2@WH2){W@2SxIAJ(BGi7EmHeoPkIbmjHH#IOfF=I6@YIARHb&qXJ!!QuX-w*a3j`&7lDmL1#`zlt(HNDVinuO#kj^TH=X*)z="
        "ua4yZaKAfmHvl41@ZtbMn=c+CJ%;GtN%6fDyIF5B(NXzfFr00bu1b_db%9vQR@b^FSrmKf)HAmGr<bDAM6z1b+E_|c?lr#v&dZpD"
        "fZXT4o@}6)9>Qr!%%bv+$<6yZ(ture^p0*n(T;R?Xk0ekKsq_@iH!or^a17aA89(n!R&3ESUSIV*Y+#4YP85cUPaK+53q8A8G?^|"
        "L26v%GU27=ALGjMm;V<kkGJh+nm<rW0|XQR000O8001EX=dStY-~a#siva)tX#fBKWq4t2aBO9BFLGsYZ*ysMZ*qBGb98lNc`tKr"
        "VQ_P3Z*+4nGh{R|VlrenF)(FiW;HWoV>n@9H8N#6WHV$mH8(V7HDWU`G-YHlVKgvhGdMFcH!)^qV`VvJF*r9kH8MFiE^2dcZgr1s"
        "OT#b_$KMb39gg@$VQRN-cJ8ZK8Q0`OGt(p_S8)u#yG`363VU@V|A+hCWw`(lk%AWo5ZZikAL%(ne<I0`Yq1*j1`{2n?>fWTwyxBs"
        "&{|<pm5tnQOWIY`G=#fiTfE54{#BAHOPtf%SV}|gEx!OR(-?$++~&TPY@nE)!g)%JqV$f*jr$a7K&6h}(fuddo*oXh%ceU>XU8qE"
        "k-(Tfpq&0A4QDtQy|oil=lAN?eu-B07TL#}2s-)>W==3&@R6@bwX0nwyte#foLT<z|3c;QzFZ9R2T)4`1QY-O00;m803iS_E7hIh"
        "0001s0RRAL0001GcwudDY-MvVa%FIDb7^#Ma(Q2KbaiBTFLQ2TaC2yHbaO8>G+|_9GchtXWnwfjGB!A7F=I3`Vm2^2WHvA{WH4nk"
        "H92NEI59FdVm4%9Vlp@|G%z?gF=An2VKF&oGBh$SYIARHb&uU@!ypvL?+bm0gS*+F?Us+lq_47!qUK2KjEb7EQugkPNy^xuR|oxn"
        "oZq=_Hvq(=v*ZXws;@YDe0JWRQN(>tcC*e|z>~1UpklJ+Lh=`yl{Di;MXBTkd#$Uc5KT>UUW<}tm25a$(^6>+Q|>Lj0ItiJ1jk~Z"
        "`g+t3eK@$wl9)wdO++*A>revgx|22d_{sE$Psi3I(*uNyi9JRoAhPfQ<?`=wI>W&1Z4FsEzjycc3#P@WS+xExg7kKPl@rw9tdBRO"
        "*0d%O-dg@Kt}K7~f1%QN-)^S)15ir?1QY-O00;m803iSth}c2j0001s0RRAL0001GcwudDY-MvVa%FIDb7^#Ma(Q2KbaiBTFLQ2T"
        "aC2yHbaO8?I5se2Vqq{fF*#;6V>K}}F*7x0W@a^HFk~_`W;J3sV=*^0GdMS4I5{>sWH4o6V>CHtVmW0qWH~WpI5asfYIARHb&pGG"
        "!!Qtr_krADP&X~4X-bM^<tlLrdW^7ASrYQNG=$#0mhDhl$ZC-O$9!{LuK-xa6xjiobzd}Od`UT;QPcCDZ6;mG9#1-Uougt?l~7#0"
        ")@HAEvKC5Ls06oM3Tq8FZ#Bn?i=A20)_Ls5+_Ag>uJag$B&jWZE&7DH-=@o)m_!`|VVL$Av>=T+#ek2W(hvA_XhJbPKz|9;A~_8t"
        "`w!srf5!2Q0F$=~Z0`Kt+}p3E*MpZ7^IZg+V+RW-xGu#^H>4&sp%C6${xL2rfBAo*%6MO|#`yzKO9KQH000080000X0L%vnVc-A&"
        "0Ez(s0BHaK0A+Y#ZE$R5b1!mbaBp*IbZ>HbUvqSIWO*-hZeeh9Xm50LFEuz~WMwpAVP<79GC43|GGt<8VlZQ5G&nIZHZo>8Gi5S3"
        "Vq-TpIWsk8V`eckGB##6G%+zZVlgu|V`XDFIWB5*Z*FyuQA@)x6ouah`wy4+qA+#VlC}FQR>o~|LxX7&l2jbS|8CNDh{B!@$-U>C"
        "?_RbW02atQb_AyMSDFGn4`k${n-bg2;~pcOgd0bl%GL|-D^*ubD^b#Gy>C(SUi7HyTFC?F>d-W;sQBtu+EC2CxAFqGERqKv(@^qq"
        "o2L+;{CRQAL*Z;9JNHRafKr~ErTb4b6FnR|TbAx1o^2W^X#ol22ke*sK=}#_^KNI^lKH*6)i2P9$zXEfCV~oXgq0KY=v_!xq|SD>"
        "5MERMF|I6s`G28`yl*#Ie*jQR0|XQR000O8001EXGdt}O-~a#siva)tX#fBKWq4t2aBO9BFLGsYZ*ysMZ*qBGb98lNc`tKrVQ_P3"
        "Z*+4nHa0OZHDx$uWiw+rGBP<fVly!^H(_NsFfukVGdE>1W;rusGB;%~FgRp0GGsM5I5=ftW@I;FWi(?jH!x&3E^2dcZgr2{YQrED"
        "$L|Y$hl9J>pzCUCE$gc+qgXu>H=|-s%1YV0FD5BtgI*o<|8aijx?BLTj484MFzddk&-jvZ{6J058@3wtlj`uOW7|3^HnkH%Szc|W"
        "thD3`H&Tj|FoKKSUT$AaCD80GpVQWP?1tQ1c>!FfF$hUfQ~Db833InjmnkudItIcp?lWjX8FP#QAJ5A7_;jd4F+D(c3Dh7t4WznH"
        ";M0G`;fw&Iw+?LT{9fJLua(!mR}}MI1e;?EGbgw<#Y{J(I@F;M-dg@K&MbfVf1%2FUoM9E15ir?1QY-O00;m803iT%jzm!40001s"
        "0RRAL0001GcwudDY-MvVa%FIDb7^#Ma(Q2KbaiBTFLQ2TaC2yHbaO9aIW{z8VL35mV_`L8H8VIdIWc8nG&eUiWMX4sW;ir5Ibt_q"
        "F*Y<cF)(5`F*#*nV=!bfG&MG2F*jvoVPa-3YIARHb&pGG!!Qtr_krADP&X~4sh=XvRq7ICd4$!<l90xsA@uIGY=_c9R)h3E=9}ww"
        "1Hb}$$Bw|1{-QD9ix2JtB|R5xH|wp4c#>`yG$mUm6&G1n9Z=-5tm`5#TF%Q1TiKPBs(8s?^FvjyX{8OuDfgCM0M}(qf)~`Mz8;Oo"
        "5cmGFBxX@MOK9hP9V!rAcXAdV&%%uObZl%gJwUuz>XEbr67du2<v-wbhK1SNShjS2@9yoF!pPAGa^Wt53T}Xv6ZGI+pc_(S8=DAk"
        "E&mu-mcRVJP-(nxH`Dw9P)h>@6aWAK2mk;8Api}^5s}~k004^t003zK003opVQp}1WpgibWpHnEX>@OLd0%sMb!2%jb8ca9b7*gL"
        "b1!9NVmUc6H)1t4VPs@uG%_$TVq`fnV>dE0I50UlHZ?eAWnwaBVK_B7Fkxk5V>L50W;bMFVKgx|VmCN7WMnRCb8l{SkKIbcFcin{"
        "2YZJ@yiu6iu5*oh6)Ph>IiYcB5|UIL!*{o7J49huhvffpe&@1Y0kA;cu>&xrztRxs*$4MQm>x^Ene<jfI-=`)ow99@lFL`DN?!6U"
        "iK65$bybzBMxwe!-WK()sQ7M4D{Ux_xwrfRxXfb|yhvT{>&SQt@yVa(#3Z7#iR`pbk^&*+(OJ6x6lS1@Lu0e)4&vFSj*>=5BA#G7"
        "{|6e+urPTW%jVAS&8__?3=T#l7j7b`;Cfg%LHFK;bVX`xV>98k<sak1@|XV?Dv$T|YMeh%O9KQH000080000X09Mc2<wF7h02mJd"
        "04e|g0A+Y#ZE$R5b1!mbaBp*IbZ>HbUvqSIWO*-ia$#w1Uvh7Eb1rIgZ*FYu)y;0(Fbn|T`+&WJkX<&UPW5L>>M6VIDFOtPL^a+V"
        "$MDb4qS(7ncDtb~iU7F`IoJoo7HN{g2is4ps_6%80`04~x?`(UJq%+<#n55Z6nu5VRvopcSE1XsJAJ1xJO@4Yv=67_u^Ofv0bV98"
        "t&12*Y3?wB4<3`&DH?!etaU-c^k}Z*YdY?8|ACIw%muzx4=i9E#u|rHxA}8PcdQy%e>jbGWF6%>8a5n=^<&%C6-D+F>ycPnk3Bmz"
        "bi>BJYVPNh1Jx~co4F$n;V^ZHRZ6TV?sHANeNN~%?IZKAZgx#k6#s;G&3e6lle68jLfj~}o=)5Cu3?c9!x;B@jL-8_U+(hibbMm{"
        "W0OZ3s=x;9e=weVOo5jT&Dpuh+a<PUV%JO?sH;FtPtz8@wipgn$j!{~xA6W&DW?qGNWVUQcsfrkE+xiNYfM5a02x3aaRBXg)IygA"
        "l#5bRG)XMP6S24gVtI~Oyh98j+USB3R#-{Mg)NOzBAGyBe6XGy&6U*)@kA`IfLL8hEKHO<rHI_9EUFcihP)>%x#ke^7d%i@3K!yu"
        "SX}|JK1VFyA(kQK7gcB@yaHpAS6EtbDmZJTw?U(T<orTB6YDD=E-xkKApvr(EujdiETxbhKymURDxEYaaKh?xA)biKD<Fn*#OfVl"
        "lMJ9DauN}gPcAu9Qd)}n1@=~y(j+2f7UGE*u7KEFO6<XT<AsgNbF|I}An&p}y8MGSQCmY`9FGg}L~Q;C@o!K|0|XQR000O8001EX"
        "1iVedW&i*HYybcN8~^|Sb8=%ZVrgb?V_|G(Wo>O?FJE72ZfSI1UoLQY4a_kPgD?z6;T?%PSl)@~T8X6tJpk;H<@}+M2&tTu^EdPR"
        "^qBdTVo^EY$XQ6sRh#{^rE*bKiLspFNg)}uP20Lth8+kU226&pBqKwGn0_2HkHa{4f5U1&cux!Q>>h6J;SW$t0|XQR000O8001EX"
        "^CrSQlLY_(j1m9<8~^|Sb8=%ZVrgb?V_|G(Wo>O?FJE76VQFq(UoLQY#aLT!+cp$_50L*r=t&xEr6~H8!2_Dc>4LRMP^a5~ED#v!"
        "$l@ZA8c8K_yZ-kbl9H{Pm8`%pzu4m8xqas<d!Bchra9w4V!@S`Rjird1h|%)QZTMnNLBzO5wSWt`F#H2G<yH}{B#)>N$7dr;lben"
        "D@q|Xp>kbPsbG8XrxN_|AeEvZ3Dq>tsZyXwb6q7Y)-}DPI?LHzL%YI9BiYjKSSn(guc0Vth(@Mt-Q4pJ4+wsizh5msoqt?jTt^pg"
        "KQ33Z!SU*9`K~#>`gD1A{$YD-4LX9S5+p0o>H6@2<_v9y79}dEmh1^H1Y?F3B7t0mj5E!U`3to$)^;YKV#)3x>XOIgsF<dy$diWE"
        "L4NY#^4%}1894*0D+w1ue*&6pmQo67fI2)#ASDH5+{Z9H9Fv?WeS_h=C4Z9(fze&quYk;dA{f+T=cbL9L|PNO@U2`|#@j1Psqr-*"
        "=R#@BUU(<Y;`helZr<GGBEDBY5-td>v_Q;QOe{8-L61BaG*O}F#0Kt5Rd;BCGbN-+65)Y+Qa)<c3ojO^hbP{QNcgK_5|YKWtf1y+"
        "T$K=Ol!1&^K-o9mtM8zM4Dxc}tupaQ;tb+@o8zRE;sH42F%VQ_u+PE3;G(GTyXu==i?j-9jhgSx=evwSZE+@83~DjRP{cE(Rjvn="
        "n^(6%HwTA~c@1gq%3<r)uMLaNz_|y>oS_5kY3_pHk~s8wItE^B1?x=-3sa9flg*Rwxf|+e=dTJ)#d|>$*FxNn8E?<GCSNnYA!@_X"
        "Vv{NoQjlQckP7)oWpZ>!<9qXi2%c}YhFc+Bt6SmT69xZ=0ouoJZfC@!dF}<1G3jRPP&ukh-~qL?)p>DQ;j<T432n=_=d~=3DfX}z"
        "rnw_;(YkQA(<;wVU#T;q4Ss#B!mjj}<>(NdAu{fHQug=3pe6RM5ftqC*8NivTRo7NicRXCQ|2^=40|T_kB+?mt~LP+ry8%#hBRih"
        "1e(t?Dmf@MYIegsu)M^k$E>@mTa;rR5gFC4rP?neOxfA61!!};s~DDArm*g`WU<)w%+@uV{CJ4MyJOS0+`H$X%Tr3$DbVI)`bC&j"
        "MX5%J=JR#@*cN#!n8((tP>aZb<Oj0}HkY;$=1<CUP{=!>#3|x2_D+d6-Ol1S8^n#(T9ZS?zQAoTEemmO^GsjHeFE&%o^Zz+n82ig"
        "HMur$LfE${+`6kvgxSk?3im8HzR@2{hQlokDCwpaWe)<=(%dqEgSN8--A+lvGO?8pN%RP8ooN-Bl>-t~a7Gc&J<R(&XFS3DTD%@1"
        "_afbTgnX==`ZIf^tW44F-#q<Vq~RW|uU0jEb?VS-(-yGzy7kn+Y*T{M!cJIM&1OR|>TkaV&${8JchC5}>+o)z*XyjTc(gr?tT3~^"
        "c94J^+YWZi1sUlI{iWVa@Zqh-@*RlQcr(3*mu;Q-&os2z{&-si+E~^O91m3u0yTn%Et;9L-wo5?g<)y7g`VOXdM#v}^}*g7xq2^Y"
        "g{^L!qcO9xLxm9Xs)%%kgDz2ifv+QPW;Jo$BX*qn_i>*J@&6vtHdNmgPPg_vTV?hd<7@W=-{k6Wt<J9B**-CIix7@Qi4H_6)q}|O"
        "=p)cShvSgLy&;&F{v&jWrT7+S-wn(zIlR5Kc64w%S)$TqrPYA{StPgX*t1~PH*su*&5G8*`P=dY>ofb=9rq#Zk^P_%1rDp$?uMPX"
        ";!QSr-KS;+e2Q_gAAQruX-mNTX!%kvdGN%v|9VDJ@2kD>?GN4)=;O;_1y;_CT(0bo;6)P8xg9202p9rfm--5D=2nuESMHAp3_=u{"
        "`4UAo08wQAmWZMz0GMxBoU4t}P%MoO`>%y5iGKi4O9KQH000080000X09jot%})RT09XJ303rYY0CRF<FJftCZ)0I>XJu_|VJ~54"
        "Wo~pYUtei%X>?y-E^v7b$H57JFboCIyWkEXcU{1no0R5L3N|TeD!RSC+hgX_rV4}vH!kb&Efr``)j;tS)s(vZ8waV8quZ!q{!+Hq"
        "lFFduAUw!3AUiX!%Lh<P0|XQR000O8001EXV(>{Ma{&MVnF0U+9smFUb8=%ZVrgb?V_|G(Wo>O?FJWh8ZgekfZ*OoeaCvQ#O>5jR"
        "5Qgst|A)b+1iaq#>O)~^lio@Qy_Z>O<Q3HyGjikn`;4|5wrQiAKzio=m|++Wcb{hXMls?}3K{RbftquGf-F8QfXM=r$!y62n(A|c"
        "kaLk~7>2&<JI6V~(YtIgEDbc7oGR;60Ut}|3Qttf^Zkag1R5{bM<(mfypI@S(77D!8{&bmu1w!^eIm3k_sp5w^u99zMc^y_rjQHq"
        "VWCv#z5_j$9PpZMgn}D^{>-2Ml@oFMDV2)M;8VSY-RJ)lUU==O>xZ>TnKc%)dm&_V+Uv`@uO#)7k&NK{rjZ?2yO2iFa%{oSR`sJ}"
        "lGD`Wzj1%G&=*EE&m&YHDX*7r&zXNoK|?z+YSF3_GV8gCxa!0swUh!ybxwA=G7lku_)X_Vt<#Ijf>j9Wi9j$CU<mnHfvxfP5;Zb9"
        "8=2l&zg`%<l0K^CrY+Sl!tf=hq!7?Xh_S)O>uwQ`fB9&m+s=@#)otIs0#Hi>1QY-O00;m803iScl}$BB0000=0000Z0001Ua$_%I"
        "X=ZO@VQgn*ZEayMVRLh3baO9XUukY>bYEXCaCsHU(FuSs3<bb{!5tnMzy(~w6#@+!Y?G1(q)sp1?-9|VW>U~3xo|Sv>fB+D9+JWO"
        "=OcPvIpBfxTTEXGbe`3@jE#ua<pWSl0|XQR000O8001EXmZN1%;{pHxVF~~MCjbBdb8=%ZVrgb?V_|G(Wo>O?FJW_YWpr~dXJu_|"
        "VKiT5GGZ=jb8l|#R!wi)Fbus9*ncQ;ZW7yx({#rH`xV0=&=&1zl_gK2-Dbu9`zU|aak}2N!_XiI0=_4b7UlQ!{PDxh&8o5zwNP^P"
        ">1I`0TWZ{vSXHpzufHUp*NRrR9nagw<tQ4b1M@^w`%l~LsVPfk$^vrS<aYaNf~`Yub&J~yl-cIi6sp{O_tp%VTXd>WC<RA2brL0q"
        "bsVjubif_9%BvpSVUp|*adAlTcps;`c)u@>AhWv&ciAyd^8Eg|PvLG@+SHyLLM6z7@r892Ixo23m@nO=n`k4kjfNG3PbB&t8GjBj"
        "jsg)H;x!^u;29`Dd%V3urL(ndhzZdqx`rPF{t)eQd1k#J`+%mOLNdlBooftDW`(vu@nJP0Xdu%lDvKi1Hh)-V#LJBF6o0DGkx+(^"
        "cS0&46kh!dMNFghNGp^9Ewn8a32zI^p{{wz%-YQ!tJF4Q+6h4lsVYX(c<Ox?P13^@RZ?<OkdtbGh7pvg8&RLhy8I*v%xSft4;}Kf"
        "mx%%*_>qK$Gyf`6cY8mpsEi^7nt4v}5h$@AP4a?YZ^eNR)rj|r!yv-czxL7aHfSYzOT;wWvFX2r6+E7QR_wiz_eTE5M*8@-@!hWA"
        "Nz~S>PMWdGP=-X|J&$*BbT?^4?~tPyciy%a!6zo6GEMbTrc=a_<v53X>Gcbx4alIQ4|o<W=mw|tOX47Uog@W(m(J1a_3;JV3V#B}"
        "E<3GHCMmC3tY>~P#a-ovVbc^?cL{MFP0Gqn$Jz@HDx^N&^Pl#826e?;kYwR3fjbMnI<WM2vYe#1GhSDKPDl_+N8M*~pBZDAOC7A&"
        "g?gY@Bk9*$=j5`j>vC@tlUP8b>7`EpXDl}rdsIf+$4hG@h2qgVd3of>HIsdlt;1E})f3Zr_Iyt6Fz=uB;(?aT9P$$`vWK{|O}Icj"
        "D#^D9T`w4|%^@4zFgYYMKYW2+^WBabn7kkS{S8n{0|XQR000O8001EXxU6!B;{pHxCkp@oCjbBdb8=%ZVrgb?V_|G(Wo>O?FJW_Y"
        "Wpr~dXJu_|VKiT5G-57lb8l|#R?BYNFc7>K=syHL$F`(MPMTwZenn9bD{&>Up-7da<hViqy-Ph9S$1<P+Cz-M2zF*4<kCC)^6|sX"
        "&8E>DB@=w}c(bXrt|aa&Y#P|6+b`+ot#F&$foFAR3S^y;o>@EB+Q<FA?y5?ts)Pb}h2CEcus3M6>~Y_KQ2RowQdGO|R;wv;k4BUN"
        "IU@^&PNL+njiYUpO}N2cSTTZ#5tfH2xjQ6rQI=Vh^84%|IV3#AD1G>pK?Y?C$>iF#))|C`IRna;+B9e^ql9z3Y?tk#9mhVJJRCkd"
        "Hr7b^i?@*#oI~Zvhvfn+0VPO_w-+cky49T{LbQu6;U|VaSig49wPj8}p&Gl8%y#ZhE&3w!QfhGVVKXDBAeYD*jVzbCcv?rq^N4Y<"
        "KeuQc6Urgo1Q+0tS@F{+QJs`CtrQ9*le!X)SzWpU+LoG3th>drDyeg#)d-w%(GZ%&q4tF}$qvCPIH#hZaH0n{wV*=TvG(M&DQf0G"
        ")teQ4s*vJDR}e-=ViHdDEmS+){UW1K!U>SXGmM@<g=1@ySN!rS;wVW&rZKYbgm3%T(dvH=QgFHx7MeR({SUB`diP%x@?PY7k?%$R"
        "-6BW(w|PD_@XT6m#h@?P<j8#>Kda)qIJysmZj||1Q3iodnHRe-ZB^I2rWwY%kMgjDNByuXx$wv&22XfmJ;)A&|C%^W?qRHYJWZD9"
        "rT;nHGh2h9RU!Dtp}>|DXCZoFf}6%NMas{x9UO2Q1$iL3CIJBh4Z1Pj%b)Ij0S(EF*A-^819M`u|B!wS+LC0q3*L~x)#zXlmf;g5"
        "bitVV()#_u_<SHYGwIh`@8rC!%XDuf<5)r`-D{rlpE+Gfb`*;1<GH>&f$XRaKYw!0bYTZ^y7l|c)fdxQT3jf&!D2k?*@4!~0*V?}"
        "(Y>AP&Tm0S!D$te?Rp97bpd%;9%x@o(@W^u_ApRAlP-k*egjZT0|XQR000O8001EXgY6oG9RmOWFA4wvF8}}lb8=%ZVrgb?V_|G("
        "Wo>O?FJW_YWpr~dbZ={AZfSaDa$jv>ZfRy^b963hb8l|#mfvbzM-0aA19=DI>p0cF(TsW-=yL?kjO1==dxO37q9N~IEoteJ)+L2r"
        "gf{GfUCx{%X+Hh*&3<_K;_B-9_ISL#i|=mZ{@!oo<_CIn{r%O4^HME7#?ObR!||bRU2n2XozfBpx>y8tW*^Pg`9s^<M%TQ4_1nvN"
        "xQnMhg4LbV@uzq={1T6^-#vW_iZp%MpxKR=$OPVvoEo?#Y?!J#kheG0X$XtNbvvlF+=!vSC%X*(b{!w`aC?!XHwhM@kiO7ywYFQU"
        ";Mj9X_aN#quxV_ZxC8bO8fdNSm~CC;_$+d?^r$*(-y=!5rvgGJMA12#Fo(NAXt#Q->K(wVXGXo&d5(+7xA72<ek!d$FMkJWI_};("
        "kG@-m?rh@Lv~63ENXXeIW0rC&MFzC$jSj8lep&we$K#5pr{kX_VwyXpZSpN^kvR;aUHfX?anO#gDv}^*u0*32P^?i&X4Ri{{VB!A"
        "^Vi2$f60|R$CV(uW1n5D#A>t?x!I<@!aEywZz9|LtK&-DcraQ|0NB)tbx7nULdZ6gYEz|Uo}0o79;K!b;4I`46rSfQsXNr5F)g;7"
        "wze@4h!O1238sZ7tOT>_)u`VFm<8491CrqX#1*}{Ih>wBi)*q)F}1}oY>bs__?{i2rf<<IM5KBDM`*1zHao0f71gy?`OBDTrD3-0"
        "xF>eBYMPf>V(n6;Bv@0a&kV>fLCaDymXsMSjkR}<tj17O(C}s**V$BdNd%Uh2C7aoVk}lt=1a7G^wZl@+`qr`Q@nn6czEaM3{Wi{"
        "(0PqMyz)4JQ~)bSHyaAtxX)Qd11ckMqt$7&#pPqG&GtOJIB6vBLKgv2S2HR9ode4m%U0*^vmo%EGHPZ@rBT_V20m7e6MrGRLZovB"
        "BlfZiYCK&OtP@Nhv~<Y|DapOS@_<%|*KpWiU*!eymB1^118)RGohRigge$Xw?B0qd+7ckcnl~>h=o`HMBD~DAj1}q-;zcnUYk4;n"
        "&9)nt_U=w8bwunLrBj79?wm6WsQ&-Re-qx9FY5XgP)h>@6aWAK2mk;8Appu<X{n9@000C7001EX0047xV=rQ9W^ZF*Y-eR{ZDB8C"
        "Xk}w-aBpdDbZKs9E^v8`l0i@7Fc5|BmHHn>Ie`?~-D?heKxz>Zum~isXlhTI5wRVOCtWSSe#c3H-7N=lNjx6=z4>0Ms(SqPy@A}J"
        "kcR;U11Sk)62_iiQ8`7x=%F3^9yqf8RUafBC}MQ9s;X+aST0&g!LHG+=N)huk|^BU@_lc|ZF1P}DFj@F<f+f|>#h58Zn!E9w`JqA"
        "gV|x*rry6D+7jw0beHsj(Er|uBq@^1g{KxW9_ZP1<X(ovQO_p5=2WSMH0m(wvu9~0+vjlc1MX6!X1M@+Y|8Celp1g;w!B~GG>Rh-"
        "I&d~5d9CS%BEK5L4s%Dj2J|=>(>!4`c$mQ;TUBw!R&EyLOrOqiRwAlusIJ4%Q=q8xB^Mc6f~=@iCgixE1boRaSWsEnnXfzsOXvK9"
        "hLtqQKR`Y{{~P@T(J2R^-N=>%itAX(mFi?fnKE2%wm%;>kOt*YpsZ)Vvxn&?`VxY<VNyQAcrpfl8ok4T>h}O@_?6^{nP%i;G%FtN"
        "Z_G!u#y+869W{;1A5cpJ1QY-O00;m803iT}!ltQw1pojI4gdfe0001Ua$_%IX=ZO@VQgn*ZEayMV{dL|X=g5Qd5u?FZ`(E$emCI%"
        "!0O4~Y}b+{--ZlpQn$;{ra_zx1A;)H#Uq=GE}lf^(#-#ThZH4Kirck$sL6A9F5kJzvaGYAJtqN8(u4;zA~}ER$0>t|WU%JB$esse"
        "2|!pH<Zs_yo&A1(y*RtRxmpZjMrgvw>CO44_x{J5^Y`*<U|CkX)o$^4n`VOiou^5=6=rEn7!@>#Xr4n(G?nIz2cmZ%Y1S<jn~1MU"
        "wo5!}ig@1gWL*+IaRC{Pq`uG6<R$+K<fOd5wHD5))g#uzJ+<y6Q@izBrK*GSzCv<xo$a98ZmBzRN`w1Vn!sh2hCG6KyG8I{j$-Ge"
        "(okG(IAf6b1&t@7xQ*yDWVvy1|KM+cvc^@up&2m$759qf;Ail_bDk!1g6J3bTpYXoX9wA{As$_yYBN#{rK0$Z$0gD6#szsDrK=-!"
        "&vvxK5oZXq|Hu=TKE5LB=Z5u2ofNbnO$0m%KjJYLCCwh$!-;<UfTpc>5k<=@?4)gD$9CO<Wg7viW@7WoaF*^?1rhu>+Rn);O(SIZ"
        "K%*Rr2wz@(`EdSev78f`Tenrzdd=Y8;k%bYtV)!5lc7@(D~8_TY<r`zaU6Lj&G_vMjf2IUSs0F2t``ocHg#tJ&SW&AQ*Rc?KZngc"
        "X47$KYO+S)IPTaE#~w^)s25QRr<AQGLl~{5ffEF?sq4{+`KSSt%5hEZ_DYxC|8v=`xr{L|opBwLX1YH$t#@skQhTzRxHC#;p2I?W"
        "<jt7F0-%%8%$`usb=i#8Tm~3II37A9%!wcjy`jq{-V`&0xiEC6Gmm;Sbg5$*KJ9;~zRxxg+;3CNsw)U}b*pCQb7mCQky<<21F{r*"
        "#G&0l5+FVEWfjZ5!=N~G@q_JW@wA1E$B>BS76NTwLSrxt(MslZU&z1Y0>h}Q1dg#3G%5s&N8fzuw_$t?*&3KMJD|IqM$%;YKydI&"
        "$4ZAD(@eG;S$vu)qdbuvBZGi%O|gvQmGJWrF~EW|&)Qg=F+7T>P<YqtSphFwrN5OxH;cXSH6$YSWvzE|h{7)E|Eel~en_Yc#4bzp"
        "v?;iXaLC7Ka@c7^w?}^J9e|c<mDpq1vn>h-%_<wutrT`wCX&)zduY^HXsy$99l_f*#4+u={TKOmZ;_u#$Fc|yOCt!N%f#wqCUuQU"
        "(aHz4vGM9z$z==?Lx$8?D?T%cUz=d$bXk1{mRriRP6eESXO)uGokH|z!fVj#v>IxsJd^|OqeAUP>)ZEVw3fZuO;mg}OBK|j?3!Hq"
        "`GHGin#opG$>FMm=j5bxvX0B*ugk^N`RB#OwZFVvoGmd^v@Zo()slk$m8wdLTdglmit|m1`$}2^Nb93cHstA_Un95d9K8YpEx7?e"
        ")f0N82FW)iyT>HEKPb6d1MTA7=Y<4K?63!Pa?~@B9g&1-M&t=s1k%LVNXaz_A1X6DRWto|`h8EZ(NwT94#r^cmE;52NMy;TGn;a("
        "p+9M~gGH95Stqn^lKUimOvu4T@*+jO$@T;68)=FU)aLur#AH9P3_l+z?rc<m;jAB1k@2UF+@_V!I3(mMtz1`d-qr!=k-=bar=7q("
        "UbQVEHb4UT3s@f^`D6Lzg3vYg`&=km`I+zwN#FqsAp{X@iWjQo<;5p~>R0qRN-4u=%I|`K{GE5CNy^6VAd?HJTpx>q#es5S?L^q0"
        "THUUmerRwYfE{)>WowS=V(O!M+1+rjF_$o_F`bZXj-lFyW(*@_Qz=<%jul=xl5bT4l+&mC-ORbuV)lBW4=XU<oK#+!Gpm^@)*)7;"
        "2rQ1dFhq}w{7_-4zTyi1`pXa3htD{xTko$83)`*#08mQ<1QY-O00;m803iS(=l*9e0{{SX2><{c0001Ua$_%IX=ZO@VQgn*ZEayM"
        "V{dMBa$#e1b1rasomOpY+b|IR9>{+X{^UUGwfmUS(T-+N*xIm$F<6ID>~kWb7f(8OZP}0CNwyQaNz-(O5Og{{clVs`v7;!uJUJaR"
        "p%HAsGjMDU_y<6<%xH(U%1{{1=F(&<m#|v^mYdAIyS$iPq|>YE<jd9dBvE-1MbU6D92C|lmUHBp<jxVCP^GcBF&(i&fXo}ulH*c}"
        "dFwT;EKH3P4^y&xxW<mgW??-XoUf5$LU;mwHnSCshJ#>Z-=L)P<p20tNEi<Xj2<6l%&3}rNtGdJY%H}chk~UkoyAmWfoU2$kj1D0"
        "XhGOnKgHmb_?^+lQ53;95iaF(XJmyCkC?G6N|K0}u6Ijr(FN<>U9hg9f^popb@K3!<-!5`&SeD?YmJSgP2B<BF;zLFUNQ@kqqqk)"
        "y}7;m(QFFsL=Fs>z?|l#TN+HF(HjZAZB911S=#83Jm(J5%qS(WV=bsDZp^SMCH(C7>WC#t^6StXxUiE4uF7?fsU+k&gH+_5c+ONd"
        "gRY^Jy@cBG$2$sZfSs$l2*~XMFeM_h`ckpK?AmCcyW#&1Dqv<-pv4p9yDFhcnp!Bt1BGT>THyI3yP6#r#wv~o`G+QlaFIC`9vYcR"
        "N?$M9J@g^kseG9r(3#1>l5yP@(t@@Z^<>l@WyoodV?)f#4hoQaUdAjJ84_PX31S*OUP(+zemwhUesk+Cw#?na@Avfk6xe@Of-b8z"
        "yw{6l2HKftb$SA|m$X!Ix7y{|*1H$3b$T66@;S&YjE(4^2_kYti+j;rRk~5Ndu<!B<Bu%rRvvZqzyelQ_lSx6Y7taqi=~7jH~2?;"
        "{QDKYa%hXY1O49=q-J!t)2OBPytJkSi;o+o+ls=N`&g7;t@57VZhyAfjrVaPjb6l~B&(FGRD66&c<hwclBPOULdPfj(pt~gf2Z{V"
        "iwCB)Ys$V7p1i4$y4VDjW+8CrD3I(tvp0hH1}P`K^HWc$?^Sc<(@Hg0T|M3i@j(G6_w?4}K9m=H+txRy-Yylls}8cQbU(fYdY#`x"
        "9=?}qljmHO*s}>;SYCfS(Z>SmnvnGG8jpWaO9KQH000080000X096G%QiA{h0I~o803iSX0CRF<FJftCZ)0I>XJu_|VJ~E1bYU-F"
        "UukY>bYEXCaCudYK@P$&3<U3$`iGSt&<`L^{D3P$kx5*L<iwTZ3%{pDT)6J+Xo%=4n~fUgZxjwjwjRzo2Dle+;Vs0+2`;SP$B@7("
        "kzchxQ3tvOYTHzr^pLU;(LBx5>ele6t7X=ww8qOJFWFRiuN{hcV9#^)%Wplzlvp(-oN=RnO80sC08mQ<1QY-O00;m803iTpza8qk"
        "0ssK11pojc0001Ua$_%IX=ZO@VQgn*ZEayMWMOn+FJWY1aCBvIb1rasjZ{r*8!-^Q2l5|=?Zp9mH;2%JOADrUiXlx+6WT*5VJwZj"
        "B3emE+Bl~4zjw4=e}zyYW7hJ{$D8+_R20SZ@@fny(I*NK0tshh3av=s1C5k^hG@~p+ND^+9T7+;Vl|!IPaYQY#q4H&J1ZMm7DaJ>"
        "c7CRUYd|<-NF>dbN?Z!ijdvlzoI7B0p^x&IYNMCCw+r6xD=f8g0?mq=21`bi0*VxOq2I})4zp)$yrI`JqClmbk-LOz!qf)3b?G~y"
        "+{{AADDp=bj1i`oFjAUqSs0(60k7!VAP7D}(OM4DD@(GQF~5|gKy&N^ZBfKQB%_AVpw)_E8h7&yA$p^E!;D_Q=+mBpUne8{?mOni"
        "JYKgM>=?YZ+;WG5vjtjbHBWiUz!26t)!?2e81#Dx<qm6JQwP>1NP(GPeVRrmXedTpQ3Q;@xms!Nwrp1sT!1B6Q8ySKAz@fi%29`M"
        "tcSMRu0UJgW@Ai5Y8jWJBx?S@I`R?Qpi<iK<XuASfUI>OS)Po2*l_VkuZlJ}1a(<#-eL%<m}eduvI+iV9m*|de<6yR8e~em&&|=w"
        "y_?|_VtLg6;@BH^KWBH$R^6LpaO#zdUb=U1lkEqR=h}{3W31r`jh2}AdGP7wdnOFxnqsb4`pt(uV+^NZPIJ&z`O<)lVblpy<zHQL"
        "ag6bEC+Gsj<MsXjaUN<e^-4HN)z%E2b5$x{4E|-)ytP~g#^F`}JgGF3Zv)3B_@<Q6mVgiMLq7K`X{_`?rhE*Rq#R#O$&FBPcD_hN"
        "4biZs17`1HZ~Wf$Et{#YRavQ(%7)(>cMPefO+Hxs4PleF>=ITelyK{So=LR1?{Hf&&>Dlx=i|rMtf)u6U7X@H-muWPyGct}1SfmU"
        "JNt1=Oxl<ujr(i!luO(CV5$Ra9P$rPO9KQH000080000X0H)DR!O#H!0BHjN02}}S0CRF<FJftCZ)0I>XJu_|VJ~E1bYU-Hb963n"
        "d396Ej@vK{yc_U85I*e!iMN+tQnYA`ePAzXk>pk^0!E^3wX!5gbg|j5FBL!9Luwm_Obutq5nWZ){k!)YXb5?T1oX-g6&vrsB=8}x"
        "(8A!@SU~8JRYw}|WN`**YK$XYS5;LOWzj_6gGNSWF(pc1`rsq~4K1LPX}m()Qr}~(yt^nb_#FkG4EuQAI}EAwd@U_0Y#ML%ob@>N"
        "vGu$o)aQp7eawPo@p&oWMrhv1eLTd0Zp-2lgM<G??#uo#5u{C7fLzXNI`TP40C)(Nei+9qsO$P=1ZWD}UE9pJSTj3%<ScN+#K+0&"
        "bTcxfslCmWmRI1TX^kAgl<%7+5iffGsLJbQ*1m=Hr~h>%$MiVlzy=PHXf1+F5l&|&i0E25F-e|f;Ua6^1fe?2jmT;$qRuU5${h)x"
        "1Mg^=qaZMHO5BShh193)pp9$AkzkE73>`XcbHq3B_1pd~psq*8hgXw0?@0Ym5kvPq`k$o48jwLrLoZc6-=_~CEHZ-5GoYKT7LYt`"
        "PJ^fIwqE)k7ZFXOtAjEf$OSEbMYV$Jxfg?batv0y8L*|h-xQOK{yG>ZV-Er4Q!15SCSw4tX<bgLCrO#IKTt~p1QY-O00;m803iV5"
        ")`me10{{TW2><{g0001Ua$_%IX=ZO@VQgn*ZEayMWMOn+FJo_PY+-b7axQRrl~v1b+b|Hk8}L6YbdnNDwR1F@0%`1nfuabSpclgs"
        "XldmTp-7daTRUH0QV;9tJOl`k#M$A@a(Cr8&T@9M0EVd!a*`yN_>dDFKL>)d6KEvyGfGH>j49uT@Q4VBh}D<n-P2w4uzpzmSpQsw"
        "SrR&qGo4H)G|Pq5piRKDeNUyxU`JCC6Sl`JBVjDKmL%2+D(oZvxiR;5yR8pT;srT<#pD;}gv!;CWI01iHJ!waC<W!hy<m)(Z5GoB"
        "m@B~)qKI;;qsUdrQtKm$v}KEEvH;38+``NUy5~ZofhKsYKL_*K%&VB}WeLJ|yuq|zd&oAWLAGD5>_(qS0z2!&(wZ{qr%G?ejc+{&"
        "zI9U6l1WqGLN`RsioGPE3gI_l2V6-hr0d)hZt-!EIt0Q2vs|BGXL&=%YvPf|B0(AD5|w#I<7%@Lg1OaRVGW>(@@#q3B1}ThdbS;<"
        "grD3#t!KU_g$I+E;p)4FmZAvG=+Pv&2rlc^vQ}(CwQyEt6$*|crZ3p&spuv>tmv)ilpbw9z|gc2n#etByV7I2;HlDE5*AiU2FMP|"
        "u$qnp+e5@iEb38pOjHsx%3b$r<g|Rax-7l5*r@hA_zJTRX<zgFpi5P-UaC^@4ox~Z$34;XFZRnUVg`E)89BP<=~MO9A~IH5L|v(E"
        "+M#6+t~-d9ZwO$1Y42u>mI`Is*GOp5sbNnUKrtP|^k8b<wK0C32+tpm9v>2Y%JH@gVuJ6xzAtiCIr%iF)shkQawdI6DOAMh3)U^#"
        "jkklLcNUl8E4^>VGqOXbI+l9DgQ44NVk!RlokQb?45b-)&9U^#x$?CsB7=*CxY(X$Y14S7<i}tG>LMK<;(n3`?1sl8+fk185A?>!"
        "xgMD|y_sNJ(`L0M+$@6mwu>oaEy(_n+%s~%*m|SL&EnneGCgFCZ3btyg)4A}L*)Z}-JqVbDyEg~()rloRVxF6{&{s6hDTHuYztpy"
        "XKc%QsbP?)50ZC&@lBi6#p6TMIaq%x_cc`i0Z>Z=1QY-O00;m803iTANXHGx1^@sN5&!@o0001Ua$_%IX=ZO@VQgn*ZEayMWMOn+"
        "FJo_RbY*ySE^v9BSX*=BG!%Z%uMplBc{g>_4uvHU1_sy}USI|on4z2TD7KtvYs*HGyUi~B?|i!XlFgQZOxxHxN9TO!s-q@J@*7io"
        "ZZ0atIVri}su5CXBMMUU2T`z^6s<J;!H~B?mhD?j^noc}lHHN8x+YpYkftqpt+OOa77NjItuo}m^r05J`1(e-GCnuV9OAR$i%PW("
        "bksF3jA*4Mf%IM5L$C7mk{MRiOlv;SnJz_PmLqB5S$Cpp0ejDzhGiDJVucxY{ldFe3)8COuab58(3-_!@y#HELc}{RH@~T#rwg|P"
        "AkO|jocZeo!KZ@>Yod*EZ`v?r@>)o~CPEsw#$|c3;@l|BWLwcH(5t#-7DUM_l3V|4YSlTulw5pGYJ}q!FI%r$anmb_udD(hgn+8#"
        "mR-F2_TtZc8y{8|Uu>TzOJXHub^Depnx;XC0yeI4A5QAO)^jVJUZ|XDr0&Ijs^r<TXXKTr0sp-hkXe^J5^8PM!{X?f$BVcYJ@c5p"
        "mT@ik<}q<9B2dk3PULC`Ssqf+Bhev%>oY-JIu4<A&z&RvyJ_#Z6z_O=eeS;+_K-V@cEKVW^6Bb}&pgfJb9e{p%hk9q)I=m*T`w2T"
        "cc$-ZZi9yWM3#oop+J6?^^tV_u10}5c$Yg~ArD-nutHc-@Xqk^OR{h4(q)>y#NXmTP<BM}d#>=b?;AMG*PJu8A}Cxs+OnKsUry1c"
        "<STMTT18}QJO~6ziH@f0A;gLa&B^bq?)eW&wTdP}-m_YiF$TGh4k{K;#t6CBN$R%$)pK=(GeQB=kkn35>RAdVOM#+8s%*&Z)`ydG"
        "i8lzNo(XOt6><!|B^tHMOBzF-*>xCo$B&zuHM^3L3YFxNaAE|UVj11ClL8eZ^}LxSB8RqTcpjDIc6+cb0?k-i(n0n}dq%!*om+j8"
        "k%K5pm<2<PzJ%v0vB&-b6{T$e_FQrWT_rqH9MFJ;f=^o|xXuQd9csw>X;2pnI@5q^XmU9L2a}mR(wUK+uS5M-m;+6+B(<tL>yY<}"
        "Lxx7-9_+V7FVE4?dpOv>-@}krC66u~Zj-|^C`jiHwb>zL&!>Q_KSuD7=aB$PceE<9H%?RLfn-hT-`b{j4@I^23lAnq|4p6mKfzg{"
        "t$05<3ccgJvj{?mA37S0b561bDkk@wu9i3frEh5IyqZoxxYFIAy0bT}2s>`Qv~`gBVzz&q&W#wVA!D6IN5|A!4g#*0uENjBn<7np"
        "(+`CfygzDqs={SE6nE$pA+c-$gM@N+^?Fj+uDV{3<h4xj<(2zvarkrh<J+5@An{d?xdZ(|7Q#g>O16TX2&dHUn?kSz$6@8!C6AJ!"
        "GqOF;;fp;Gp_TLvwZD!hySJ8V8nB%coRUlD$k`Hjuly;<<Q)oi=vP*XFuYlk01Lw2p^1(bE~mn6AQg@RpLk;!W@tU@2zBf^c2)q>"
        "DqGER)+$)~j++lAx!25M)AS@&t-B${>$$bNw==pZE07!CWcKqmVn0`iFtt7!t;BgZjiMI{wpDgx9ZEk&o5`NPd23&6$cF$XrhY(s"
        "ZsDI<wFu5EUuV}JtOmTc+_G!%&{cMQ;{bntgVzwO%adG7KV}Td&xYZ4wT(U7#>XGIVPSpV@|#JZedfj9FdNk9bDx54Uw{$V?U6|*"
        "2P9@ZdV!J5a-yAu?lPUx;^7?T{Y{wjezb5gt2x+@!?@upB@R^R>IDuC8ul&;&c|GLyb!Do1Bp-klzcrMxSpCfth$^(6r7iOXiU4l"
        "#C0X~B|m~pi&EbSc*!oTh9dR7EHrj_W>h*Pu`3C0wQFR)m7N=uXR$r+f++OAJ7H=8a^ULeqb^z>6XO)%7{>w9DflQNC7aD`o}Fpl"
        "DbDVu$4T7wuZKy4DHxMfIIAjQJ`%`M>@j`-X}C6KxMN=5d=}Fwx;E)eb@38IKz85{eovSc!xd#nzxM;Oq-!zH9g{Vl+}NG_#M(92"
        "H__bu{wi6HJM7?{_nMd3^orG*PiSM<K%7k5BL{Aq{Sa?3{m(j?NSq$^5I8&e=g8u*y`Bbh!V{7!Z_&PD1lBR-CVaz;@4^5U@9YmS"
        "SxvY8A|BI^{EewH3|^rA)Z(UhQid^owqD+2xb7iKTWSB}Fize!9q3c{y!l17V@F)3T|1yvXmC}W0{yvP{ny*XcG~aWyIE&4+xU6d"
        "X?;w|CmY8V7M{FKJ$=mx>3mWRgE;%&l&tKVPDfo!g3Rd^&zUZS*!Zl4qcN*9_)BkSvb00=+7?g9L9c3FOaA~+O9KQH000080000X"
        "0Jpx_r$8700H9F-03iSX0CRF<FJftCZ)0I>XJu_|VJ~E1bYU-Jb#h^JX>V>WaCx;n{de0olE3%8{tr0j?v_eqTJl#NXVtvKQM$fO"
        "?X2x?+oyP0iUcjz6v>iQY`5$G{bmLL36PNN^xdf+7V$9{4CWg{IgayY6(w~rPdSTMX}T5WWwELlhsGqkW<j(}sw&B|S1jY7dCB-4"
        "4_7s>Se38Jkh3t4xaT;|*I#}8RkAGdvSvY5H{TaQwMdhB^PJ?(*I%nVYaUA8e68~vo+ZCSLtN%dSRhS#SOW?blOvAvRaWzI$RZvG"
        "tF(@iu$H5XpvLvo$dB-dm{H$6iSsZ>ulaHrcu`OXULMD3l5xL^Yc)e|+%Gv$A{w%aXyeDw#>=8ivU)AFk{5ZE)OmT&nq3vSpF}*X"
        "llmU<xH$Xa{CMi0Pv2j#3H#&guNZtBT(bzmo49t4R3mjH5i~g2swmU#DGd4ugWg{{Lv2XOKPPxTIB<0E&3rFBJlu)*_IMbKU+(U_"
        "jCmC8e6zo|dl-ki;pku#@5TlIM7`p*pC%#CDvofLNv(!F$iGgQFP92Szqqd#v}7OId2)>`v8|6^i*Ow7AH;kY@Y)@P2T>60jrYQx"
        "`FOk&0`Bue&ZE)cK0o*~2xt7RModa8n#&|C^D2+)?Nb<cjN{%3xNDJpxwFT2cH>|t4){*A6V3Mz4#VB}FdiR{#{2Vw@j<Y^`^|X&"
        "3z2>J`m2xA^Y2eCFQ@PP<Fj{D|HH}g^z>q?bBH#_34o{zzP00xnx=UbZS8D@X|RGN^v2z8V^W39_b?o7jWsB%OkpX|!hb-*NW^`T"
        "%F4y>gCjY?aZawYyadgWm&>3=xuJ6hey@hW{B@9IEKNXEgS28n7BL|UaRQ2@DtHRQcP*3$!b=EP)NUI;N*j37Qy|XC>32hG-@EVM"
        "od11#ZtFZg`~Lgs>7}jf^n<l^`s0U_)2XfN<n-MdQ17N%&x80=m-k2QP@u=E%vilDQtr0RG|;PRP%YPHD^QhzUdq}<u6YKuteewF"
        "a_@pBo}7u30UZ=XsL}=|s3e9p>idGTEGLkX*6k=cf!~F^sF~X=d5K@9WtoGzJ<0I+0@TH_<>DTm@91H$ZmT8tKzl|(n!2U)>Gs9H"
        "XaD+i`>DEehQwKeL5I^Sz%79J++a2dvagQa9((oxE?{)ZMemT^CQ-eZ94f$KCNNzOj_7~JE*7ZEk+JivNCPl>aT#0_m$RT0B;{GY"
        "x?TV<uznSX;d7pai)B#WupG>08Wf&n@|dMO6ZEYH?5}Khr%yOAFu$W9*6zUAecRqnK@vzS>s5<jXaB%;K8+mj*E~_n%o29Q681uD"
        "{c7L=Zm>ajFz^=qE=sO>1uM;t#`{-8Ha^gJ2U##deklVWxT|BnC7=P+v78|34jQMb7^>7Wv2$dbyH;mR?}8A>x!~cAPznTbOP&(@"
        "pCAkaw7hwNj0S9T&*or8Krzhk*~LFTfIjhX9R&Bbq4IAlUMAp{fU;xgsN6FN&|^O)HGJJF@;n76CMdyeh*kv)gN#N80@O0elI3c-"
        "g|x)JC;{Eg*@Caiq^gtf6|49%0PPjRY220x`amVL0CDnFU4dp}RhSptYc{HoL`L*|0AKsQTk$j=vgboqFG^l5@-&)^yu)F;V#BVJ"
        "(>&wofaJh!+(*xbnsh-!XkG&en7H}g8Pf-k9N%l~%;I!ZppPFq9U72cUy=d&YtrMIe+C<`2c4189LC_5G;WTK4YQ;o@xnnz9hD|r"
        "tm-#3FuRJ@>O+$!<o~%kz(IeQSAY;_wYR*9lF}6q)#S2V0rB}AoV}ml&|_zAy9J4Z7v@<8uBvNU=o!0Qz+T|rfJI3Kjv~3-35v#0"
        "y4C?m*)5YE0|UtmSE%eGzyi1GS(o>eLmrk%QFrVLG2*-PvybfZ&D#%CcJiK0e>%CiydY&)y<*lrKn_j$x^7~MSM2zF`sQ*fdkLJY"
        "JY9A<{ppf@JU;=W{4@Le^yeYFE`w#o-hMcHJLpU$aBidYEprX9-RTNh7gt~>r<c?3zzr~_3^iyptmz&M*pDZde>?l(lAWLZc=GO*"
        "W72P+i*OJ{QhL>NX}~nbR7_cyD-20AesA_Sv0Axyq@dKqV!8&p(g(gF5vy{PHQH7MDtw)^Uy)?3(F8mNHAoOlL6i)>UIsfn=sG9h"
        "x1L|JvvYRx-Rar+6c6M~95MUnn-4#LQ|rDSz8-+>Z;!ks=%6Dt^E4;<B64~dOt{Wp9WXi0KOFX)45`>SXjFp`c2TwAc|QI4;mxr`"
        "l-Rh3XnWj<&m*wGMBLk?&`ue}y`44t^IgHXEv3jb6VIbJC)i;y?6I%<ibrfaJUSW=^o>;2)>E;}Q@k@%gRbD5IxK`A3zkRy-5GXc"
        "hU2e16gLfnRt_Q32p3Wk{~zS<NsM3Sx5$?rwz3@i#q`7U__9@D@6XS^m-6uA-=@F@4IY!%rciXKKJa1=`vz3pPLDFufYWS5a*o)M"
        "Oq;gV!UJlq#tYY6V*-;dV)O>82ru;y=LU|NG8Y&ZrVvI%KFx(4LP#BRFtH)V!qf<DRkBRd0DP)@XoQgT;Y-zFOeoP<J5DAg8Xbs`"
        "JID5archIG9VD1um_<ln77G!O=`8n4nGj_wk7INe8V`LmB(2zP5LDygXf&`0UAS0fH@X*C%_OuV1@;Bbz3NY;C<whwUg8Fvp$s>i"
        "*GerD@UE()^T=<jCy5Szy5ciDKKtSH(tSP<a@1%~AxhE%V6J0__QJq>-56)UzB@nr;UjzdbF;2ZBR3?p^jY>zk(A~)z>(9b0XZyZ"
        "L%@kYGzI%&BHA`ew3y|s3OrE@SsP8HYi3OzbUd-kP|c(qgduXfhue)Bo4xw9f<S8tC+3St+U@e>lXxCK$}9LH77}`d*hW}+!7~a?"
        "p?w>qEeEJ8;F(P*@_GSIJmLikRCp|_tmm^(3{8Fn+umk}1G`gUIaqTEs|oJV6O~FOQs*$IahBMB3pib~-soT$Y|PyjNDEa48^H$w"
        "fo&Nh`$;Da=(m$jv})O(G9YV?QkfM>7$|M)|0)s&>txh0IL!gC8EY7#_Q5h-Gf{?#Y5kNi%_~lRis7MriC-;}=v{-~^wI)rqf&*B"
        "$b>cNa355T3xOsjA%vb+_!VWoENawF%Od5HZ(t~iy(4-Z;Mk_A9q0_2^cH0pt0b%H0Eq4?pdb<h3~D>wEeJ70Plg2vn9+X*>55Z^"
        "(QR}3KHW-X*S^S`R-NTb?g=lkbk0Sth5^5_>*O=fn7F*a%_2vCnuFjg9OE>&t|l=1<PkjbOx=OVg=1teq!tHfIw)!BUw-+;eSIXx"
        "{kymye5#&*%3SaH>%pfC`rz?DZEIh&dl689Pw*^}cw(?e+SJ7bz*_ysgzY?$YQb`zT(9y~<tJIOs(pg5wlL>$PW~~qde>#XD%>#&"
        "yz?oOCN*ExFi-3guL~8GVK3d?zV2_=(OJvBW4VCbvbL>f><yZ4FeUPuGU9+lRpePk6bjNbzva=8Et5NuGD(AsQq|kEpC(o78L~(;"
        "6b4Em)^#ACL{EXYNl^R*aZMV#1uygLrh-?)1Q<K*QDo4n2(c8L0RE{sEs&-X6vdToDDvcLuGz=Z`PXA&!GekvfD-n{InowygEhFH"
        "JUY-8sG~n<1GUyxyoOE2_(|yg*x-ulyA-$1lsnFHX%uTX6uK)}w0uE0qKpA<`XX|1x^yX&0+ks$d9A$&DuP`m*GU$nOuO3!3IV-P"
        "d3o%V5_jkxfjbw_D3%3Fl?3kFGY1TxA15i@1Xg*vLVUc!@GybFZXX?kaKapnFSu1OP6%%oh&F&YOz+p|frim~VDi=%`a3bRSWeG3"
        "IQAP(>LTJ{htaw!U`ZY<(I7c0<Y+I@nkyVig8D@O=N|ducq+*q@@c3Pn7j<|MOsy~K3BbIj+Y2NAtT!$HVrD3Y3~p7t8@hn>#BW{"
        "Z1Bn$j~5DJy*5J+UD_S#IS(RKPvq;r5;5;~8yQo@fxMDHj;`>6Eo2{sIR3VRQQ$pYqjTO~m4eH|3z1`=T`6As6}&K`2|?*8`ig_w"
        "goM-^X#<FZ2u00X8%+xJ`-+$Q)(eaALFc-nLmWfsC7}@$qfElR8>({=Nv=B-Yvc4J=GZAXU?{>$3|9n4bQ~@r)AH(5tHF3w)r*-v"
        "%=EDmrUv=~8-rRF9Z!tr+d!RsCq#m0t0i6Kx`=r<6q2Ez1A`rk`^IL!d9kcM8c$7Z{+>W#B7HuMFq4U1E+MiPFvA+W18B2EW=-7H"
        "A~TfBO!PVzZFTTptkop~4>9OTA+~b<^4pn%?B-l~Jd3K^q+YlV>Q7r1#D4CrgGY?(SXPtz&PR1Gvu}+Kyi%scL#L$d`sfl_1A=7E"
        "HKia=Mx69?16XHU<XU}2TW$u03-!mE)l4x}du+wkKCO{qH0%XvGw>H@!&i9!Pec>e;cCrnpwpdcsP&H}3^y4|V{73qh@z}@IhQ^`"
        "RiM#20DkKL*!Y41JlG?SA3&_F^c+yZel?&-i}hoepK_gTbXm5*weKiCVULKQ<Y^F1VHt~bJb8bu(bRS*OBQU~!^O$_g2i1rKj8~F"
        "SMoT?(SYMzg8};7pGMN<j%-Pe^|)r>`TO(x+Ddl`oN${=3|uxWV%MblC#dk|sjPYz1>KJ-ozZGyh8p@qcFw_<sZ{aI09|a3-yT7r"
        "PY3WfLF6JIe0~VSm7=O>UEmrjPbA28!18oK&^sQJ44OCW6C*S+`%4aP^ZqmzB<-aZK!qj0&qV6y3gaQOy-n^|RDt?fHiKRe-7NOX"
        "iR@t9jst9P{$SYkj}1w%B@?ZBgOH+}tMb#HEXLqHf^tW1OuushdrRbM0nYIMHeCxAc%EeBIv9JCAt?OO_hfNCu&`#Ld6Nw@3Ak(Q"
        "H4J~cQAXJ9Mw{JXyTJ*TwKTlw9$#1p#ZF*fT<I1K*#?Zebfo+Db&#y?)B40be%2&D@grSP2$VGcbtW3GW?0pFwI->l&Gv6NB{qfA"
        "{P=LK{MEW@)4y@;ehsV@Sa;Y)n7W?>0CmS^)3h=dQFRU6m95Jd)5uNHsU4#LoJQcx)K}rI9%)*q2pMbDMnBH(sF@A0G_gn4^s$T$"
        ";Nka8(`0<<V|+>J28upg-?x-*WW35kQI6-6?V-%tIs!gCN#W7nR36wz2L2+2Ow*-MDBq*Yw$P3Qc$xxMF|`*I1;h=wfK^Y5&`gcC"
        "#%IzMXqDF{TMt}TQQ`}A2MyY=tcvnU*@&6asRsRn_fj@=#42K<+T9<mLQygF8lK1e4X#5Rjb9%6_dF^yLQb|Bz5W;5AB{%nl8r2z"
        "R}-!H2td$61%%>hps!8|4y8e`-cTFY&2E_2B{DOoPH|}D=p-98%hNJuLQ4>nNFu(PsW-z=6*+uCz3Yeqif-;lWMqnTRk2&GR8<zj"
        "vTrHj!rbwJS)-~7W-Ya<gIB^H_ll6Q;I*<B+|oLST9vDIcZXq9$V;l+Z!X%j9363>FUub(Si#Id<;uTw)daRPnvB1}mv%n;aT$<0"
        "fhEUAs;tP#07J}5tRz(57S3f2cuJ<{SyO)vu3>@#4~j#3nLs21M(nS~g77%rcv+%+YkiUmH$-hH-lbw`M_Uptw>cOXbxK(D?k0L1"
        "bh}7^-U7-jxxxT7>BvOeIZ@x6rtbZ2J0#J%BxffI74%uObRXT#le+Arq2wMke85?tY=AUCM~xHMvs*f9H0uRUL;EN76a=eA)QV8L"
        "22%&}j2#OmKu*V(%@zT^84}=2I}`xlRNFGu$Cj(AR+ZRw4gt}Em!1tD!B_1zWO7?y7m<Vsh$GJHS18C%B&toDiQ|wbk7r0<ar6vk"
        "b{*3-xBSg;yoeXw8poSe)yRp1FFx58T!U-h>d7}1+Ohby!seSsI8t;Lmcb48qol%`pxP(pphO5T;iX=<YD3EVj4&%#cX)svlQdY)"
        "qd?Tm*gOR!G^HB}a@qARhyzv84G61w2_=GQKx`S@xnrc`3do_0H+ycV5*61$o0fAY&r*A)MP+hYZeAl{#Ib@YkT+ilNm3oYBDp0}"
        "s0y1_QGh{vfDs0Qlrbd?tV?+Ou9jz9wTFyqQeZ7_=V6tNr_7fHc#N&KHx8JR^4QSk<Xu<!+hQGvu^?dOpB=IV1D!D?wboQdiOeq8"
        "M!6(8jQGv4>H3`oE;rO^@mYK7L|aa4rzzPc)Pt{-9k7<R=ayN!POzhO9d#wBJ66y|43o6!j6gaaHHY8rMXb<4;f4dECN``pGezi0"
        "X|39fhgKGpO~*IM`~Xh;7Tv*0ywo5MM3&hW_tZ~!#euRn=)%CGx8zGqB`$*ktD(en^d%iid<o@!q#so`5r+uSr8odfiqOjoEczh_"
        ")$LusNRJzazCn~ugca|W)zB;tsCDCx22WTGFJb#@fy{te3ToJum^H+AbeGcW@#mQL;8pLoq4O@4nCkT>+A^p22~Q)D*pu@x@kbLt"
        "d3$vZ5gR!rn@heC1)3yM#|UWiVl>zUulNIL<I8u)blwn6K*-KZX#<Tt=(QZ0M09qYaxU0n0h%u59fcqYRk45^@7076IFdu@g-3Th"
        "+Svo`jDN78&gs!!Af=KoqBzx18ghX4I$XO`QEmp23bBKlCOlj5B;Kb}fp*08r|0GvKLxkCp6rbF4$a~Hj(JdVdOcnA>>Z3$kL@Bl"
        "i11pDB6PWw<u3Z8*o1sTZj~j!tvD%MS^N{1od1rM9$0TAuH3g38epLDMab60?!N{P`Y}$fm3o-|^l^HAg74n?7aym`7wcw`;=?zA"
        "wHpZSjePYgs(wKs$gfmR8^wuetfC_5>v}=%>(x;H_b1$>jLJ%SHR7@J?AbGRqE*9Wf1q*ti&jSQm#c?Q(2fJ&1Idq=7=c_*UE)#D"
        "9U8QnWnI<Oe2LEc^XD*{B4Wab5=Hb)OSSYhs7!LNC!no97*@2Tb-HIc8`;uxrX1)WQzk5PB$JPK$D?m{)&tlyto!ni;@dw%f&PZ~"
        "fQQ5znQAX~)UB9qM6?3TD)X&(^>O49(gnZ+_0+0K)bl533Z?%f68b!aS5fjfxufaQm!xianMBQ7Y;1$vcoDWqwh4^}Gtp7a8&GP7"
        "Y(}e-X-r~^8hoUhJV?Q+WSBk?HPU?m>UC);azk7LJWw7cFh;x%E#42c-5B86&eSpb)19}%Yc-l}_bJ_pdtzdP?PD`&G=?(W=3xt?"
        "gRj(?DiWlL{ZyYtVtOloYl=pH1_lXYlEJMw(-5CcGkeU1vrU`X5j$)0l?;a3bAJGSHeJi;A?A@Kr@iquLgiIuMLQ<x;xxbAcrhVQ"
        "iR#mT=j-kl9#6rqNj}gH1UhP^hG8wjvoUA><TQBa6#jUesv7aywE2VSq#H?TH=$oD*pQCaQqj`a!?kG|EZ5}wg$h%T&6>ZhFl%bQ"
        "5-(zTq+!RSjmg8tA4d`0U`;#7hzv-ZCb^r@oU68)Mo#UVYJ-LvZ#64vFz4x!Tx3S66Wo+|COwPEjwaDzQBQ~lK^Zo~WT;>4k07ug"
        "dHWRR7ccke>7{ynDIQ<aBj_i19CsDxgW;OnbM)E~=_wFB4ZrTlHfHjTJbyz?l?5QJ0%MyFvjP@6EA!`usG~LFhFa`K>>a~xD8lYo"
        "i)A%cpC!1biPfs$<>#d0ku+T>L8~QZ%Q{j&Q}$lVoONla-2sRnDey5p)8_p}J^PRnzh(2DNtx%h`f&y^V$gPM{vC$*0jMpNYU+qv"
        "lX@afY;sPNLeX`<{A~!rVw<^Q$_wUr_={wzURxw{8TCSjsi2v~5kj;waaMJ_b?D%)H{2rKCs-$8k75;eDzzoZW6{3i<=t+QOJ=#&"
        "6iPf}=i--^w0kA2G9?8ds>QG<9<^r_q!`$NH^k?8eglq70H;}ccEAY4sy@_5L?s(+Ri{YZEEn4_uT#6OSDrk!oUEJIia;M*=*crI"
        "HWfcI5kc=$xX40Yv|1LGsJR6JTY=sPsxV0;^U8`Ey*`)}Ht`u2Fo=7DPIg6%UItkbi}Z@F48`wniJxC<18ZOYehA5=|9Naw-_x2f"
        "BkpQ3A*!-`p{&g=%2R<a-dA#hL)mOY)?IKt@%S1Jzv_$3T7mBCw+E9(Ge^4{SXW%221{_RR0@u2C!OF7#OeCz{%by>cGznCkbdPL"
        "t0z0wv+t`90nJQCDPUiJ^}kR{0|XQR000O8001EXe)w6WHV6O!zZ?JnBme*ab8=%ZVrgb?V_|G(Wo>O?FJxhKVJ~E9VP|e{b98BA"
        "b1ras?O986+cprsXD0svr4Jz!iFVvhr&a4_I%(SJr89kAWDWz7kcFBASOB!;xc=|m1;D2$S?=Rj&LkGe-NimX?1DiM{KB?oE=6%i"
        "6r(v)B<J)>NNsqgzaTWHm0>xllr%Dv#f%7c14MGY<{762Suv4qN~*4j&ZJ^G4uatMv**uvSxIF;Cf+=Mma{d;U=>v?g+KBSni{IF"
        "b(mB5%gl@zdCdg>z|@SqoRN}l(@Y8j%BA*=zWC(q^G~D2^JfIEs$La5n}JAvLk%MrWSPw1pL@HOir^xI3*zN3gX+#~qzDo(!AhaZ"
        "Xv^C@1H*tEpWZ)oPf=n;&1OcdQo^8U4F<A2#FdtTe;y<;H>}iQ6!jLuYva6oLDs>gI5{~X-|~Vj$UTaExD>&V+N-g|w+S%mx(o{@"
        "!YXpQ;QChmh`b{B!ikTDQY=LbK4c;fJxOaq)FWQkMFFb>LHxVqu!@yDG@px{Z87{TFBmS*L|rlk-uE7GZ*-81E!VPE8OCVjsiot9"
        "(14LalEjcoGG}2>oAvy&U}VdRffgvmiCZp6!>$deo38E*Lm@dkAVje>Q?N`=&Xg3=2?x&|C))CZ*d}r#<9wK$6(d%@wgFf=5eqNM"
        "F|-;-3Hj)vGhpOnYaHd8k>6-hvv*2K6$U?n!+-p`34pGsASKr&HQ6Q}>(7NYs?H3T0;~vk#e+1XI8;BOBF9)VYKYk|!nQOsV1aPV"
        "In$ZqD*%lZE9C7ANKLH-7gxl>42SAMKgK&rkJqc#^G;5MGJ|AD80{&o0PE@}_D9XM0qOgM!E>m=o#j2=ke-)!;xyqGj8Y3;O%8jN"
        "UNu5zfTVe<D~6!iAlfggW>K^&@v8~=8#&-9;m>4Q6%0_)*Dl*;@I%Aj3P6;b*6>xGUoo?r8EYb>>6{WnqVOtwk`An_%$-+YXMuvP"
        "O$s^0Z!Ajit4;3~owQ4iIEp2+u(guJ8cZ<-7fgrO3{q3%-o2yxN7)VUbeWOG(b7)HJMY37LFzdz_?3Xf2Zy%B^aa6<4r)f<LPBs2"
        "g@*74Z)kzlAR70_L*a<IIR_mPt$<2Vam7MN47(4Y52Be<9c_ws_8T9$zHhN^TajOqR?Y+`UK!By+-<K&x17Hw^HxC8CHFM@bhUIr"
        "JXkf&(od?J=KAvD(E@6&eYRWsTRdphnI>k1p2FkK)P8z@OM!Zq_<)R~_`%zQ!kLJ*OOzvm-5CDLiCnfJ|3A<N13f~nr$_oY0UnRX"
        "m2EM)Ct%MSD`@ktCxz|_;=jMY;q}#2=m1_mKHsI^7rQ;CoRFW5tcdSU1cRAF!{#PykyB`3V5)I5%<pws5oxEHk{g_IEab<N&S`52"
        "6hA*ZJ3C$_o+I3^xo`g69I{y4+iv&;{{JF*fP3Gnyb!cx@MP4SxQQpG74Ak8k-DGOC&0E2^-g&nEPA@k_Rzat5l91eLyM}Y^{JcJ"
        "g2Nmz)!lH7?G><|pPhd)Kl^lk{_$QeC5v{Iy+pWkrX}PXQZ@}HtYrJp1AM(+7LD%i_A&T01mc~<KFpC;62fwa`uR~g{f?C-O<gaQ"
        "7HrEid8KHz;aQ6LmEOW6C2w_l^LnrT&2*hMV>igRU@*b(Fnlzg0NsH6-{;NW4OQEC0bl$3XfxetF+L91J#xsD)Ea@ypE@WAz`uXw"
        "Aa8A>2C{qTV3)?WR^G;t5n*5-tYP?XRtsk2{vmQN{e94Hk$c;dm1U<F9Bx1)%$Ew<d()qgY}Yq|*^fi@<m53zy4@E4Eo$`^R(J=2"
        "1`kxVv9P*^=?}105)H-*Kw|}IK`}1|6Tko;p;AL@R>=&-vzs7Uy4-?1OnO|0@NgohpOVi#45TIfAjMo~9MkeN%l75+6bfe0;4_t0"
        "T)<3_Vk#<`F|FYS!#^qJF<2bdnZ*-7-r=HC3_~mo-ZsOa+Z(sW&f<9FU|5M!qH6<@z_+@tV7jq~e{9XSF-+FB+UyjM1ehmW)Hb9u"
        "a?4?3tBpSyz;PdGJny7x9qR4}+X<irk!W*`U@I)hc-ZLR2A-!~x=k*AEXMYLr=Na@aP^<&-;-lJ08y--D}~(-cmU6y9yRDO6ctjH"
        "Fxq|qe_J5Icr>u+k;x9Q{#?K#qCaL+EXHZu9y*08c>n%=_-5gnpg*g-Kdr1cm-^+U2;-M;qDui3fPfCEfca-H4-d&k2VZ_|`#46E"
        "tMK*EBm_!#q={~fmDkY1H9kSS?yKIP{IPRwJYdhZ)85H=i1?||S1jkYxOC$Y55M<nIS6{<_6U1$jJyvKp64N@z3un7yAb<Ljpq!w"
        "?Q}ZWa|PW(>Ep}xbnO<OFyIL)^tf9^*cR-;Zu){e>I28Ix`UTf<dDtdZE@P1)&&}<t+)tgiAIxycKs@@g}fEhR*~<&=?sH0qAuJA"
        "gHGV6)<<<dHfU6e2xS`2!gndrLTkvF#h$jNwww;J4Y}NWIILnzwj2vcB64xDLkm<L#w1Zsr_o1}F1GG@GAPHMvv7k~JnJ{k&NADj"
        "IrDnUba35%a7t}aPM~%`VVEmXB)sejtWRO(+xPfk1+43PwU?zn<OALwX^Njejut{rd)nA!#b5S3fczlEz=V=(4O?j1OLD)C9rKrM"
        "<~*XTe`cF&`>X%5;4t^nMtJapB+UndJ7MT{uHk(FBeTKn9{4x1$Y*5O?zN4>DKy%esZ-dhY#-62kd;sH15(%Lx!CKN{0gSm6@V|J"
        "r2|>zzHi}Hq#;6c`_LX?gCg4Ej5h%??=MhG0|XQR000O8001EXkfo=vUK9WT$v*%9A^-pYb8=%ZVrgb?V_|G(Wo>O?FJxhKVJ~KN"
        "Y;0d}Z*OcaaCyyJTaVi~vVISc|De^LgSB^U`R=|9{4f)AaDv&HolP=(4qzApEm5}8vZRPoU#7SJ`&Jc6iIV6}&+f}HkhDdSMY33z"
        "uZm^Yb>CM>;v{jvQ^7+gPDNQ%K^bSMlV@2na`rq8jwe>!I$tYRu(XU*?yxj;3Z7>oF0<lsEV8NyptTC)(sNz+?VGo6;!~a#rE~Z#"
        "&g+LSB1`Ls6DyDPL&=ILPI&!X6-g5By@G$OxG3MgiHhvx1X+@BTAOfGzjs-cmb_3Mav%xA^vhpp2KDY(9S(7Nh*-e=V^#M#M)>51"
        "*?pR1EUdfl<0uPQa^R;E^FmfKFQ^Jed{H9@Z{IlZ`SiD6Kfb&1Ki<6mG?FdvvXKAnju#2bKcpdlP|Y-+`#FmX9{Q4HLcQY;NYW>c"
        "%NFE@AI1mRy=o~q3w>4;*}V_{^hl8>ap?zHngVBlZoz}B2=&`CyXC2$7a5X+w&B^`Wl@yC$@(HH^#(=B_OOTuxW<dTh|}`y=tdUw"
        "U|Ox*q+Bk^iywy^B&d8uKL2p@^Sl4~>HCjA`tLvd_2!eaah~L=-D7oPDZgV$6hFAvn$z@ZA0GnX@1*XMv*6Yp89lDpa(1zvM;8lr"
        "8SLka{eE>ho9!<a;fk^SY`x@B5G^h)TZ8h)@|dNQ64*VS;%A&i^kwGdtN<Y_b@!qA>jo^1f#Y%6Sz%h`MKXEE(hubgzBQ*=1k;OX"
        "ak-yGSBuM4w3yG9QNY>NGF-0L%i!WFjHcmY&02$y!@p}~1s7j&QSRTXq5ytRKHhv9|J@zIny377HQmpGt7$lchy6vk*w3RyaK+e~"
        "2h*!y9!!@J!@fAJ;H8I3>mu!E^T>Yv_*e79@NmB}nAAO%WiGBKlS5pBVtGM!n$R&Nm6*t0AfpKf(i8PpOtkh*=F|CdJYA3H3)Oer"
        "w9rDl!*6bV07WJQ3IozNr!LsNC#SCjFF-d+U;?d#o!D<#Iu=0;R*g@B0EbSN(_noW&i5Aqo3l&K_|+=Bn$B2oIpu4%T29&JlC2iu"
        "gifM0txw_)*2{aKg*~HSzkL(($ccCv9R1pS`ly0pP-Iy-a{fG$s=0a362XVpO2gqD@D$A!^&2+L-sv`sivgTCY(;E77Zo3Y)rdvu"
        "XSejII->~`cree(j%0;%%O6M102zu}coz$>Vk5FJDLe5Y7=5Rsiylb!NeuMDYQkF}wbM&%-r=WXNIP$(0fPnfoAo*@ugGMSk2&9H"
        "X>_f<<d!z|-uit4797!@78_#n1cW8B(UPY^|99RFo5x{?wcxDUJVfO<!$M%5S^<pm3NqgVcLEAG8*az5UDG>`oD@3ZAdgPEO^jB{"
        "(bzJux*OT=9;DtERRnf(Kn?P{^B2cOYf;9aD8n|5wx{*rvEXG9^E*C(@1tgp;hDM7_$T01X~N+dfyv1g%hM{WB9(6&W!YD85*o@9"
        "qvIrs;5kHDg^%dtqWD0%Bf!(J1NzePxcNCt8x4Y8pnHWGL0ASIsUAB2>@?#+sU-uu@f}S$bpEiZX&F2(SS+~n50+H?1{`rQaK8ue"
        "6vcPkfJw+^KpCZgZyZog3ZjdSanC_u03?N{SfG@8ZdZ<^yGK!aGR%KtT7&6o3Kj}%6T1kb%d7Az3}(?RSnR=wPuG_V9)|NNTdn6;"
        "(6ovI#`tm(@?bVwh3m^%%d03S;Vu{Z;3}G~mrK5m*mSYHV8QA#0^)qXh%WXQ^TleuS_SN4&zC^|B3RGE)rtrEXtA7!b4~QwBJjXd"
        "1IG*+i692c){aQ`S=<PP=B$5Y;^+rQ2EZh|c0sy<rand#960_i2phCy#(Yo%SrVf!Ldh}2ZA<L;)k{~Y=RV1T+gi%n{nb04Ad$OM"
        ")C0t+7Q2jxJ!s>dkqIqKZX%2V3JI8kK^I<`yft(Xa3#S)7zV%x&jaV&Y&sCj;oyaw*>u%9Ixx!vq)msVDSh*}s#JG1)S0&5IjM#%"
        "+DOzBHXV>n@d&q$oZ@M9;)tSUx6WG7CnJL_u4fDVw=Uq|Q(n3|SOz`sAtZPNE+!S?dUrQ+tli)=vNQ(3YjoZ9=QzjQeox}IT#_^j"
        "_0%kaV5UdCV|kG$Uq-auHdBZJU9Qm5Py;*`hJ(m`f<-=GKgICx%d;!Vs0q+}P{8J8IUI!)DQ$mXiEaV@7v_T@5EE?Ns*J{$?$BYv"
        "0bL9eZcL5-8qR_+8{q-jLqi%=4tR~dwhDY}<szAYchg2uVi`SjL;z`+yH%P;`)map^dvWe+fd3p?1&j*5J#a!oJx*T4erij-@3e&"
        "tdh({a+1F^zB?J`z>Q`W`3&}=C)08{o9=41as)6BP6;H&qhsL%eJJM^X~AFSvL)5)Cpv^@8$wbmsON_$5hB<9V+m61Tsyj%r=e=x"
        "dEkM}rJV;>&GG*Q3|8%zWfK^o*>w7%H-5qPC@F54<3_fDztp5+>ZCddW!z2GP)UlGcC>?#9Vr7Vf+Ju8W2q;8+VcWPs#o5Dm$Gf3"
        "Lr;pnAx6m)jwZ%04l{p$ny)DAX`wbzg7rTc#P~y5fLp}14J1{gHxm#8m3mMt2vj0w>R91M<Ft(8icmu3(5~SJi)^U37+~+tQ6e7+"
        "?}BAJ@sgpogXFqRw!rOJx<5L<=4z$d%;1tCrr^ccOgJ60q6B80;?i?|Vi0IKQNl_POdfVAg82?(n3^jaqKv|b&NQn~)$gzgredkX"
        "PW$*!WtD){1IqCFI})TvzAqCUzCVEdxI;kN`$6H3x6|<z8%MjRi_voW{9owJ;UA6zKHbB44~M@?H(T<h_3|2z+8PSk{MVs?L19@`"
        "XeeYO${(ydJ{&1`RDuN2O246-+VdQ%^tR`Zl`atA(=>N-B7`hJe+$^;8{WyjU7Aq)EBuZ!B!iUDr-mA6>Ca1MWWHXn929U+V>(en"
        "ILnLKYmG}g1ptBRt?;wLZv<2tovyX%*owB%DOg9LD8{aRFo5;l&?DG~oE`MMa1+Kr44i^hP9Y=%q8L^LWeZ2rOq3+S!=dE^W9tS|"
        "Q+hUtqSfdU#$z37Mq>B>*ha%Hji5-OKA`+MFOM=vO%S-g;OM`|+(%D<<?c_w)+mP6QNG45wQVL-ZdG!b662j$lB{;zLyTUFZ>a%t"
        ";`#|U^rtxd(@tYGBxhiNo-;!s(Ks58qrqB5+2rc*Lx(Q8E+7b$U(D#RE#Zb>lfJ+Z0W^82)W;mzihpoS%#Hr)ZEHjw)`=sVuh!7{"
        ">Grx6INl9qyfkv|@YF~tG#q<2Nu`_>{t5koUmk%U$1Dkfo@8y^Vvc}K<{aZ-{s2puQ08ZhWSmS!k4}adjS`ii5#=Lq6e^9_DNY`Z"
        ">yht3Tbvv%6pK^zE3kIKPxvw*`~(A1d6y7y+&lyq6jta^x|J!R#gDYyOJAbW=|$OOQ0VfdbyD(%fn>0kMCnokpK$>0h*mbmRkuTQ"
        "56~no91#?pr@)xx@jFMPEC;S)x&vfCCts`_PMixkw*>s2ycCj~5>;|O5*GJ(;qU-K5V#|JC^+%lzniyW;X0mH7eE6;$JJ%lnYTLh"
        "u)Gqus}}6j>&v1DT5WM?Hx4ImQ$V|&_6<j^s_Y(JaqXS-rz{?AmLWo$(U_=&rH6_gIPyrvPQ#&w7t&Iv&Xayco!n;&U1CSiKtr(W"
        "YpV8?tcpe~mKUw=OX(8}LD-y*?Q~iNO+Pf%iA73}x?`6z)g-X5-b~|SXdz0cZ5vZQjwOi&hfxQ^`>B(-Rn}bIhXNCBcb-7GA6O>9"
        "56QZe^gVFImk^eY2o;CE6d7pS$%m=H!0`q}m>-+}urq*kHa&n+@gx*BS|UDxF{%nqsju#p-$36RP`1KYbDb@rh3~qkR<X1pjz<Uj"
        "_3DL(Ri2<-<UU3gDDAej14VL~^*okN+8%7%N+es6<$xbk2r)p3Y&uEHL!+_9|IhW%nk^Z~llTzh7n$GjwUPAItp0&fyxgb79|E42"
        "&Y)q%$oY{!(l-h&IexQnu|WY}WTFnUZMb6^VBHy<kGA_H!2@aLVBBoz{I@g5TSm-hO)b;ufrBTv=X1cFQa;<Q0S1DjfqWW1w~R4)"
        "1DybTYR_XI-w9D%Wqr{hYxdSd^mA`e2c1x7nNaHXBpgyB6BCuRO?_`!puZK;27_Aq8VdO6Fdlu_NSySgf^^A;qtF2q1UEcDblwne"
        "g9F9D?nO!jZkwtJlr2*{l2Ab9u-g}tx)6b~NQHWmi5vaYLJ+bE!soow1qy#|(|}!po!HE$%gZ*o{tL7#l>d-#mg}j0)50aH4fH)m"
        "xCFpdO{L#zvxtl!Z;aVyLwM2r(j^=kBZDP$h#et+M1$#2y@T~GvsH{C@4ir>>Zom~@7^#W>`_;hXSe3=J9cY`=ZRm=pue_XBY1DW"
        "LNfv<>w%v+SlMjcn9#oDaSuIt4UM;`x+q4@2D*fe^-RJOUlYLTb%lc7m9A1TK;_0>AY*J`J0Po4AMDVCjz$^I&e%h26yOjO%;;pJ"
        "xoUqeW1WrUc+-Vq45It{IfHjPU8ayoOMtX_2|F3xYpu{<8vc?3D(^$Eo<iMCyAsG?YP(*DMz>6WRHtrBTg6*X{c);S>&aw!Tlw3M"
        "uh=y4uWu6iO>HL0m@r5(!8KYVBMp%}zZ=QADnvx~`l^BE@k>Fr5@GNG{UR}=gWq;B82oNq*QQd?^S;&E?HWXBOwkI78?|o&RG6h}"
        "Kw1I0R!Znse>XdjFyX$M%Ccrp$S@4}@o#I8c7Xuuk#g7B!!m*+5}kEjDGP$?R3$P-zIK+~_j_`iBL0orMpro|e}(jkGzs*=;zKXs"
        "aWcSU2`z013cBpO@ha!V9e~qN`f$JAyQofsVk>#!1C>JwtKT64EC67-@@NV?8cF^I${=z74qX^#Qf(dk2_!xNxjo9dLO>Xgh(!cD"
        "ajo-22;v<8=?~vI`waXdDxX$vp-(sb&iU|J8W3P{0ittccQK9?fEI-VF%@OZRBAn7iNh#=<lM}EnA|LXXw@CC>LCV`24Jzs63V>%"
        "3_u63>kCZvi^sG)a!iz%rKo5(L3BnGq8Y`88bM(jxQ*6CQI!VD0D$C!Fh2Tvr7$ZMUbXWrCzi$$7GP}5dQ2%CfQiX0D8Qriy-AO1"
        "n{?f`R~_`w5ql~6G#v;6PKY7^uw8i+OOjt|5_D@s*VYTxtNL{0<Ymo}JKJ{D739y-d)J}*+&tEHkzlH3ZG%%!K)t?8RW+j^;NU+6"
        "*C?T1Y$`pY?tk-tzubKM@YBuDpZ!n2+`RjAZj`Lilk(dt;l)IIEfjt&;d*{rjJ&mz3-i7&4i(1azt9_CtOy|5N0J-g53|7chvrZW"
        "O#vm!F#~rzrZO}LIjN=%fdhJ*hyjF`Fv69mE+)ie(9*mTJ|>HXFVmI7UQM_;TaY*ZZuTwmHD_!a&~;MnDnu(Cf;VG%h%YTx_K@|W"
        "I<mx}suuNyNeulTkEsF^=3!(9HZB5M4x6uV2S=Xd8~5e`vkc(r^J))Y993rM=rTq@xm+@)&OOKbGU2%`9C2o=l#Vs7EbYtCwMPZr"
        "H!D)~2Z@P6V_+yZ1x+4p>ctzDU*TD89;~**suG+4oSu`%fMUY4TdfPyU^<#g%8ae}P)_<DoWl*hR5O3%QH9wioEB4U_XXfwhbd$q"
        "(k}|U=2nb!xpn77)0!ao6*pcqhUD^9MY2tVSQ}`FC|YHjl=q@lh1DsS??+CQRN`m~IerGs?1=9*NhPQrF+eWAO>ac_GleL4XsM{t"
        "t84-(V3WO`xIKR}Ky%s1(})tYXxEOVqE7fkB6M|f#>APnYT^R;*9M)I@|b#b7wLg&VA+0zIB~1$+tDPQD{$}n8CT{YB@Ladu1Tqa"
        "3bOjLc%nBxg%cr=gQFrEHhzk)r>pQ;&%V|%I~5?wd&u;_2kW6p?U_;sM$FvJD!}%)#(N~D3!afCb?jswdNe$i=RCF52YkEenzWTi"
        "TQ?4SY%tI1Oa-<aUe{Mg7?jGVv(#Q!w2hypD7!7i_R?(p7)A4z9{n7O`DoJwoja7mKhh0#0+Z@2u4@Kq?NGqC(F=`ly1F8AjzG4D"
        "R602+rA-6u?&)Zi2CLes-i0yUnnY$fkNQnk<VoZohx#H&F4mI5l`ppDnBAz^O8W&~E8Ef1d7&Fg+g$gQd_H?_e{^mDN!+mnf180@"
        "qvR*J4e3)bQaKxh$jAV`g9BDsJL$ABD)gEIQFpa;14EZeof_MOM~MF$Cb)+zJ0yHUNOHU!Lr8;(bx8+Mv8)m>G@OcRUM8{p1_=~3"
        "k|ej@uXHHS&Of`!NTZK-nbZ}t??+bpSf&q=Gxxs4=@$kRuqInFfs&D(%TD7fqJZ>qfE?N0?x-Rjx27(>n#2RGwkw}KRALD~KG%oI"
        "jdRDF5m=#&J9YjF&tC3MWz`^1goqDOMe^vmR=qdEIeStZKbFKeei}b&CYhwZ*d${2s&3=B?aF!?R|9qZ2Z^sU^Dmoac5q&|bf9CM"
        "l*70GdWcl1&41fj8~6H_w=v$tr^#vZ>3?s=@3GtXGvR^RZ2IGwB2>zd>I@rb(aSV!&I%VbrbUfQ-Sbu<hHf3)=yh8(sugS#rhFe)"
        "f>x+{j{jlpY1yBBtaV@C22?L>udB&cMDy<>5$jLD@bh{Qa`s1FdWQcO=)<t*#t)|(zaZ251(tH#bF`*U(b?s*MLy@xth3Ud&7uBa"
        "(wmvL_8P9<6=!$UH)_Uu<6p9C=R+)muDolPzdG*+T<yus@JlAVUy8tb3$Y~g3>c}(wk3o#a(IS49}=3Z%PJ@Ztp&ks-#ynRntnwe"
        "Ilhl-;rn!v==+#B^L_VPJ2_0Fy?yg9P)h>@6aWAK2mk;8Apj|j;C!13008VG001Na0047xV=rQ9W^ZF*Y-eR{ZDB8DVRT_HZ)Rp}"
        "X>Mg-b7pidaCyBMOLOzK@$MP_2NZ`6X-o9yZ_>0?t6Vb4q?gP$ZKfAl!+}W9VoVVXNcvHa|9f{A00~f(>?ED)B8j`$_X|tT`IlZx"
        ")`(Ked|8y0WHK+ivc6}HmW|M&D^<<fy#P4dy|A6?_N?thor_A<lKt!ZpIEIrxl`()O*rS57nc|1p;5YHd07-yxf}lPMZ5R^|J<s2"
        "_@jZOI*fni<wc>@fn}<yWQJ0-%)-7^ecefI5gO4URR=_UzNof*{sPZYn9GY?7EY^F%S`D!(&A|a@@3tET`gFvdY#D?=%r(}{DuIl"
        "%L@jdvH;1Rl6%=j+?-9jmqzlKt=9~mG{CA;`XvQ1vVHn^%k4q@Xxa6;W**kD7_mX$$wFzlV4?uc*m`8DsyxjU=v#L!j~64LmQNnQ"
        "N?U^pJz2}A;XMYyK_#OwselqSB6}!0&Scff@l@OIeg?}Av<q}JD_f){>P$u+*@dy4(gtS)B=D<jqS&{JD1pttMb*piv{pLehq7(a"
        "DuJA*xH?M(3`Z~-Foc>D_|N_Vy$S8CQ1Z}pFR9XLUuSz!-%IpEE}dW)c2hrL{+t#ZPmF>Eel!Q!AJ#<ntXO^B_6<fl$fIQ~4l)&a"
        "p60TW-M~B(wW`ZZR3KrH5G2GXVKj0YjwEH9ie0O!zLQZrB7n?1NjNyy!-WdbwIR;k9pN;Tem-C||9m7S+7uJzIT|bxAd31WLVgfZ"
        "$n~iNb$9}11D`Sk%!n2rQ2<|RP_c!W2lePIK60Yj1Lxz__W>SqX0OUxwqWwAY&*}iwhktJT^1!A!T%n3O#h^9s<I<0(SwwYtn+sL"
        "lkR1_2+=8eKa7oK5+iI~aB=gB0>AMIN~bKs?gRp|0iK67zpH(iv)7Om8D>SnQAf61-j^M7Jei{n6bw2V9|@!JNe|EJhSc@iA{6wU"
        "ppM4zcUa1(U=hMK1YUi2$9_}{5{e#)jsZGT1w-8irr7IJY(*fa?MMmqx6gn2I<mEeh<)PlMRtyXD`r0Uy(^Y~;c=Yo<#S%%OYpKw"
        "ICVIHs%uQ|Xh?1IlvBPBY>!Lz32A1k=m#Vzb7%+{XZo1pXx+$mfNLlDwJB)Vo_Vg1WIW<;oqsHa?R>8zNABuiTIPJ^iRlFszt^g7"
        "0BVHtA%j=|9eyt+<b7F@$K+8+uQQ~&;IRn+0N5_$fhZOLVj9dm=a8xMnvV>n3qr$83x<Cd2w?mI5c9(a*=U-x4ow(*6-J+VNWsT>"
        "AVgJWvTo({fO~9?^z}onp6cP`+c->_(NIpNnHDLz-`-qBGfP^PAoKe66MW)ewlWa*clP;g&HYir017ExNzrz!ds29oS#F67MUiE)"
        "fwBy|rG%*jG9SE!HF>keXAqzZH$S>q#VS<RxqK!;u@=d?KS&sTrLU3S5<qe~!M<6KDy+haxzkcSjE_3~GC?Ric>uwVM?z<~wfXXT"
        "n0!X{Ji9Q_GKoJXw5{G7^e@5VE!NW85DUl0vhHQ57>Kp;@@<P2L2I61!Sf#aPzKgYP12+$Q$EHJ2`4bXL`Q5rwVhWJ(O{P&+_2D$"
        "e#hB8va-cv$WpI74p=`}A;8wRe+CNjF)g&96kdNyZWrJ?QPwtsKSKZ;oZ-J0&p7gZf>`!xz~5WUs6$yt9wm=Rj3{TL{~z_EsLI@^"
        "anp)JQ(@zSvt#U2@x)MXX)=jkjdmMbRksxA<{7%w>i#xPgO#i!LnQtU>6pXNQ`TAMI*JtsnF_X-RlZbxXA~GjHkZESpY#=>2V&?|"
        "J&a4zN#gn|q%5e0HuS|@GH}m9JV)>ak03UD=vCl{IJU&_;b<(V3FgmcMO<9qk*lz9Xk$3fW>mRGyP7MHM|Qhed5+!=Tt7r@N@<9a"
        "Z4hNfJbS|doU`8g4(~GN$PLO7dO@_Ugy|UxLC{R%1?8cGZ|sk3777I4!fYgbitpnK_(8xFjFZBotlFcSQH|HIc3fRe9^mIezO2CU"
        "PaF8LJqDwU06{6EV<>)tOa0;mj;X^?H6GzQr|Q#qwP@PC>QXaroy4PS`!k?PyV{t;jE^6I>#a7RatMA~{9ml)4{@ioGXZi6I#&2o"
        "7)O)bX;R&3pdF-6yMTtNQ(!YYor1bS>3y&HZh0GjQtqfHuLu~})@xXu*i|wuAZMg*bXj-NBWCm@UrtW=xLix}{?N4M+=ALd3jQ^h"
        "L~MAp-&49>M|{B-cteNj2sG=liMLr<uD=&>!}{3R?*u=;?XuEy&c9W_=pdP@D=1Ag7B34e<<e${sUo<El+SqAZs{J&y1gV8ZV1(8"
        ">4-#wU7d0p6;5>c`CSj1h*q6e)-)3})WcNbpqM&Iyxq>YC9`pcGj>v`C!8zMum08NmtJe2bfaav{KbzNUiXczZh#q#G<g`0d{^G1"
        ")o=Da)J&8R*K`2}J*@in1`YCyaIm+ty6(W4`-9LAsV!^gRrPm#)-#LYo*(&)=9Z0%83apZ9vIw@97ecO6IB&?Yqe*)Q-c|TJ-aIx"
        "W!N)SvNIaQA6rX$5$((ac?lCuI~Nc$#htyr#RhnfYr?*jIqqRwQioFPtQ;PHN4!x`gI-efo3$9GjuXfAA%8!4)SED11G{RO6n#}4"
        "&@z$cn@`KXie<5V{ramnz;@?<waEs?RLi|Pmuod)QyKclSfht>%)D%y_T<U+Y_fZGW&oaXxEuxuwZqBn=0mDAT0P1d8B9z`(G~0|"
        "y?q$}_;S&YydW{&Uu6w3Ff6l7$h%rTyF2<lJE=K>$7`UgE4<(yFJfkboX;+1Tn5i<W;hQC|8F-&zjtYp^|(>UA$3)iS~Jnc*+zqT"
        "bZl~KN(>`fS0xJWz)%!+z!Gl%d#HPrUq8%78wA1RDsj(IiNh`zD~p<FDe~y{)90_?euQ6;5IY_O<tR?lH#zP>h+ybci!eI_R7yY#"
        "^%Rcs5*jOd*P8%-P&Hi+p|Af()(nf!5|PdI<ZtXz>auuY624z(hf6iy-eB#j))nM&3YdK<Xx{~r_=*Z6y$u<qGB9^J0XWT1s7D(`"
        "bmWF{2Tl+O?H(<eVl+mS10cqF*vt;~U^qnK0zyN9wUf!NEL0|{dwDpBz+BV!h-ZsLaM01Wwiu^JygY}Jj4-cy&5WI|4=eU)D)s_?"
        "KvWHca1ql+^h)u8H1dhQk&@n%vl<0jI0h-4lJ~moUWSc<dcs@#s~x6c7J#nd?(tr8p%60d#ms-uGT=TMggL%E!;^#xWs=-l@t}{)"
        "WvvYdJWeV)X*V2$wB=heFoECgDxve>_^9#<i#AaiiBJ^?1Tf{`DE?+k1L=e=!$pJ_;P|>E4&KI?qr+<#Tk5xgX{f~#H8{A@o1l+-"
        "`vJn)hK90JHn@&&H{^i#8*pGMv(Gbc5yt3n-nVxv;yl&AK&VN9k&;PB6A9-0``3sMN{S@#HL~S1C(YD!Ai53gU(asb#?BeX%Ng3b"
        "9y31{(9PU4`v`BqvKD)Yy>XmY4W||j@2wf!9pcm_+)f0iz-A~o1r4UD-?&2fz*cBt%xO8m&wG{7PMzU>gAPI^w5bE>Orak1XM`_?"
        "l)Jq6FHlPZ1QY-O00;m803iUuN$^2S2LJ%}6951p0001Ua$_%IX=ZO@VQgn*ZEayMWMOn+FK}UWV`yb_E^v9xSX*z~I2L{nkpIBp"
        "?t|3Wa@xTR2A<Kjb7_kLZ4)%v1r~8ur6nrnL?Q)}$_118?{m(fWXrOXcHhP=Vv#&LcYge6&iR?QD);Q)=WnJzu(!&yM)-WiMAK9%"
        "7hW07;J<RF5?&U}%Emb5jlE}1i^i>tcM0eG@Zj)3)s3;98Ml74a{cJqWn)b)-DcIxx+zr!GNm;&8eUau$-3237*5z}Yvn0eyu%?H"
        "A<?BOO)jb{S=S=TjrLaL-t}*OTKo70v`+@(v(9Byba+t6k~vYz41>zhd~_qKR?eC8HfGa*;rIOTfWZe>S-R~`kdA|ZDp?es@S}}q"
        "w7QuxmXsvQWnu~GY`u))-DTl2^!6@_**EDP9Xv@mQ#uGY!g@DazEb{4fyIKB_h_C!vLN(*l&^#Z*IgrWa2schEoe4oKe2CLuo9+l"
        "xK`1{7SA`jtWZv}|KRFtYmJR~t6ke*(PWWrT8nDII|Wwy)~?;^>qcElaxS9hP0@VYRPq8w6P6^&WoI9H^emWyzPt0RHias?4tFzA"
        "6e>IwvXb=lBUt`j=Geqq)Y8qCN(*~GtCYU(TE4407F#)`ofjIiL+(t1JCGDyrKtpFR1-a&5ZDr);?LI!01H+oP70f^1}$d!|Dw0="
        "&RK9Pn0WXl3~*OD_7FU054maX*@IKHs)SYk9%@v`yNBc%Btf$1`HN%tJUss%zv7rrHbz1$1EVteF!_HA8){`5X(`)%Tq`LGX&vk;"
        "&{~2^qN*Z@3hGj_=@j~yiCMHbJBrydeE*YAI*-Vac&h@Px+lG=p@Wsa%0PF9X0q^7?-8{V_;Ya?#@N%QDke<JTkPvy6JQ$zaHoRE"
        "VE${m9c#nmtz55M0>CSE5sh#OjKuNAsm97u-GN_NUT|zXfzfn2eOMgfznDeIZr-*7ASG$aIQ0I_Ot)3VN0A`HDm_F6dDp<!fb#%e"
        "6Elu#f={Ff>~kh7h@LNJeA^gQ3Jt<$Kxn#E5L~Z}w#z7Rl)ZQ{zl_5!3$W(CnCZ}Rp<bjYgVNWdT~$dPIq4%(857>(R2ZT!IQpYW"
        "a^YkN6#(3$<geZMbM(%Wy;rp1M3tGb8LG^3Q&p<Ke%zoL-)`#7#)rDTdc~%T;wXAKpTJlAl0k)s!&k2!uk+)3p-WY(uSbYWINC+F"
        "0bblXQr{?RbS<@y2yzRc5CL)FPmj1D1?}<2^_<;M-mWL`0I4%B=@cXiD1ZYMU9{K3uHXP}8IS*5D@)p@fb=2gWxt+$eDm&Y_VMKG"
        "zh~#KKcxIQgg#h!`~KvYpWmlo3VIq_I;@_*{`BG9$DiM4r=LzfoSvurcoZ|3*@TK5sKesSo2=(MQU5UC!z{W2Y)+$Hkqx8RKEcKy"
        "G};Mef?uHpOUz^MMnY(R`8RCq-ovfV#^SE2Uve&yv%#75o3a1`kftaANy<BtH>KoEdCY_(=oq?Vpled1ii8w#VkJ<kn=NKV2?JZZ"
        "IUB)9AsxV6;Jhh|NcoIj4XNL!e6UvsuOrE`5*V|(k<dD%Djr$epU$n6WzyxM**YXfej>qxU6|C@g|bl?xb)n%Kzrq#ay~QHG<-tE"
        "El>iPE$_W_5e18Ik4mM^fgixJ<;&?0JRYdX4G?dQ=zqbHZ%zR7bNrhd03OkLliR9HO95v9PSZ{zsp;(stPzb&wYe2m^ULj0<tv$A"
        "W38eP#v4>JP99XEt+4c)YeiyP9qp#LclC%x7hs;W^GmE*Fhs}H1)MKO5X+QbPIgr&-LyUp9B!18J<|DYkz&2~ECITtJNUUzg=hec"
        "U`MqcEmIzLrM#*oNTlB$j}~{^Go&jCAcp~bO)u_>;HNuU12&cd`lXmN4%zE$I1fjz1=t<t2`UdjRzcBud@=v-`1o>H`EbV)eckXd"
        "n(Hn|w2O&C(XLADRb`eDe+K<I--)}waJl;$X?yf_doV<pTZuQ`{@w8=9B4;(_p0uIf#lZM@wM3Sgl~aUKL!u$4E9`dzwYf@@*_g{"
        ">D(WAJh-S%dJ$1{FcOdIzsE!T?3o@+{e$fu4$RzY$NWk;fHy&xGfHd0fH+)9|05(ER=l)YRNYYupa^2M0jXq*ixR+pv;X+`BE$5("
        "ZyBn|;OeWdWHF4iqZGRYU`-$rJl!XoLuna~&$?E?BLe-RLdFCB1;OEu9a*d4`U;JJ*O&-zwEKRwfLrQ`uI`!0bJ@Uk3gWnBbicxz"
        "jh2XxV8u(CoQ0o$O29b4UFuwGAZ(`<G?E`&dg%t8(pPt@<qNye1M=D}ZP?uB&!71)t3<}TyC3i~8Uo1$F(9<)%|jqXFBk6FLPyE7"
        "m+?ZQk_qW1hI<xo)ibDs*7yjaeX)Q_-w5C-9lxWXht=ysB%R5XHEk0;-{$5!Ar3CC{7bjv{?5}=e@yh?8cL}l7y4YnM()gDp2M+4"
        "0!IvIp5~NYE7H{A!Cz2I0|XQR000O8001EXJCxc_UJU>M?J58OBLDyZb8=%ZVrgb?V_|G(Wo>O?FJxhKVJ~uJaBp*IbZ>HbE^v9>"
        "T3M6ZxDkGLmHz|7yimImN1D@CN?V)ScwAda?cwom9wrkOM1mt`D3S(1X=G>qd%7C{2_AE7?LLK7_J}~E(dgqtH$dO_f69u2C!Q>8"
        "5pz$*8@^@IV@2Y<&y*(%R>@7Nye(5)WGv4<GnJKvmzKgS(=^Wt?g?I%-qq#D(D(h<uU@~(wpA&Vw_$RVXY1zuKe8;E7r~pKvR+rB"
        "jJcGrU!|hldU2WOJjQ8c$kwrE=)MODm@0*#N|<7C&ZOkhQZ!A5q+$vsu-!kwE6PWG>ntr}mf!O2mW6RyD8XXYaQeV2mWg-wY+L1g"
        "im#>2VD7_Z)_uP+3L4{z;t*?ouPo!36=jhDNn}+Up`NCm<hbTYG@K(d@=?yPI<QtFa`5_<2M?>N{18!Jq{?QNsqVeaRK$y2Cdy*V"
        "3)`xxQCrfqNS1Pvg<7F;f{MHP@s|(Zy^B7)`}O;)?|=N^%3FC~G`*i@_o^1$UwVP>ih3juh1zhH#j|#4GfO>f2F`X1d;=?Fg+G~g"
        "xkh})^54svcr={lo=4En1+vv`)SiLPMUiihuKxLMc8LVDk0{e@v6%nepE{Ip`ANDsIX_?X<M=IGu;Y_?a(sHSKAq3soSdDWLf4zK"
        "v-zSkNUzVdbc7Ln%?q%??zksW!Pi+WNghKp`}MfPbTL1DbCU3RdNxn_S;Edv&)%LMU(C<g#d&<b02wY8EWJoyAc^4e6Dj)VS_p8E"
        "qYv+{W<R6);yHhFHeWB|xASBHFYAlsWPO~Tz}ne4kLPdW<9L3WvW}`B&}&}5`nCzo0DSo~FIFE#&B3Lp&55RuOx`WEC|MyDD`FmH"
        "$<mWb7{4T#8Z?0Rc9{ggaC$_UyM*W>&g+^<4iUE9X3A;Jq^#XRiiumUbTm25c^30Rb_lstk&H!FX?EU!y&p<|#ufcBQFc(9vT(!i"
        "QBuJ_)tTT(BOfKoS#evlTP+6k_<90g>pbBptr_$OTrzbC8VyW6z;ReVuuigAT~ibMchj9^;?2InW@8YrX^maTCIs2a9j^q37z+H2"
        "Qk7gD9Ra#*>NP;#_Q;<Y8d;}A6GA660>7*ZeC*H5Jr}{mXg>9d61Rmgt@C_~3J-*T&1RqHv$s(=yZQRaeE}~cc_tq0Lw*?g;4ivm"
        "+4?wmxa8ii*!oV0QUrco$htzb<VmE3C6M!t5JoMqNc#)8e|a#T#(4ofk}E6oegm6<(F;s(mD&mbi)wmtxYIQ*9~q4JMDZccxfRUL"
        "AF?C~^upT?YZx>CHND?-)%$x?{H)Yj0fbo+wbEO;1qT2&hUvPc+2a%he1!X3B+7l@ME4Zt*=zqKKOD!1)aEU?Fi>)TgL=jp?uv3>"
        "xLdB~W~E2e0&_zIfiwnSri3e??fz8PdhCq_k_e#`{Y4A7S{ZO{(FkLS?Lz<<WB6L+KCa0Q<~d00yADeD<c>p>6gA~aBSRsrTIYNl"
        "0g@^L6U&DH&E+G@yhbJ(>c&`6yBV%=(;e@?X(KD)PMUGi5jYuP?qI#&#RHb-0qEWI2Rp*438kkygC!HnMR#a^g1D*H(o`>UYJ{CS"
        "@=PXA`o$K;XT@zaaMO(EN#oj>ctY0Y2xlik37-Tth%h!w|2n(8`EoHmoj>?i$fxD9VK$=cp)gt#jk25`qn)uXqp@}YPcL%7xzVu8"
        "!$~gn1YK<C2DX#<INaJsCnOzJ+1PZNAV@nD{f%(q^y)QPz;rrDRo2a%cC9A|nw1C|4>!qyPg_&JH7eNFbXolIUCs@?lTZLTIFg2)"
        "l}WC2;NUL(*x)s6aZal7+nKmINTlvx&Y`Oflcbn6n-pczQ})E&99R<BtghUX&unZCgcTfZ2euZx#tq_T*?1a)8WPc{HVW}A)R_ng"
        "A7U*E6-6ub9<l*VR#v<SAk8il0D+ZXt914Tq=%w-1OA@#-gb!KLsc-~&nSTr0cor&jZoloK!Bk#8PvlP8Q>VR2%bN-40jb69-sk-"
        "1q~scVl6pb*DybD0vg#(L2{{{PXdg|aO%joj*OTP?BFIbdWlW_{?ytkCO*|%xyv(^Zgl{@<=#@DX{t}?Kdw#Iw8eo@s!!ohnh<Mu"
        "?HFMO{u8Zr0UglaNM@h;YH|Fgzf?zLA`GFbt`4Y_ps|g7dD?08l!7b~$O)T*Z@ga@L+>+X&lIk}!7XFo_kRQ@+h)kot6)<tMRuDN"
        "Ecbr-@DJGCw%b}U*zLV_UasX|z?2^DTDp$Pb4q0q7o<I<A%B5@i=wQ^R1^i0=P9N)rE<e<>V1Xv0>o=naJ4VR-O^i^We#q3$zb8G"
        "x{yPN3~UtQZ!82Lz1Z<!(xdEfr3BV_w<3Ov$x9Y*IRHoENDb+`fd(Tblxn(GB7j&QR3{Y&8R{J6YEPR&Q5Qki*4I8(DNw=AX7FQ{"
        "WWuMdOnZ%)<Yw9(1Sq@eovU<q3w&Z$(LqL8tEyI1{`M$Tw&l<Z9yoV;y5Xr85$c^20=W-01J4`cjNcqPp~xu%H-!4y_rA8wrAoLE"
        "VZxDrU|?cOX=u(cH@;q;%;z^<&J$PBw5_+?U{S8S+4+O6Gfc@_m%Vg=oNvqSDO|w1S%D@zo6qOESzEbjj-UhonN6izH`u6h@x){V"
        "rmAf-1e7JDG@&Inp|^6p{KyN;A!jHz?D*_FupB}HT5$PExZ(FncFPeiK+C2M9-yw~-iAs@8FCJ;zvGcAgJ#m_kyx{$j4R?2JdZy<"
        "AHN42lY<?}5DlYrWarWXAh!Ncs>`x25+lb|b9SsXXYy)*CKK@~2RMZm*m9uIlZakP=*nuHgBBic?;su_67gJ0>Jais{;0gO2TC2m"
        "UKH~9G+`g-kjm)z*FwYZbnJNJi#7B^sD~m*=pmCnKH&2bjKJr4oWRFo1q9;s(LgkId=4{O6F!L>__WwTdiulAqbIl$hn|d0y*VNn"
        "`3LO=TKzQHZU$~5jU{(nWa+`z2opNh6Nclzzv$l&eZRxUu4Amrq;ij{pWny4QjSu8Z-|YAMR(a0PPk;98H_A=@_(S#0to1cZbZUd"
        "&XnL<K9PTJ>*`QeIS{V>f6ZNiXjcJN*N#J~64E7Tn|ISmocDm~q!D1;nnX2XK)lMyin?b*$Z42Pv9T(IIcaz+E8@zcr;05oZtC4-"
        "MKbkDJeb0?6^2?~X7msTaESM(mACdWLG&>lL|*pf?919;%;)E)r!Xh|yf{1aA6|kLEh<2{awj9$dH`^NcMZyofPK^?zkO#w{hOtC"
        "FPs8w?V-q|H&`3$>lN<)0cv5mW=--<n=0U{Vak9EKk>d<ImyLuS}N|-l*g~@1aAm1+IccuA>H9ob~56wF;g>~8~v4xLMCmZqrcu!"
        "&!W4<m7S-rvR<O9A?cSQL!!SC=@>NO^Jninef@ZJb@|b&@hqICg0lGlJ2>Sfz>vpuYzSLAPo&2%-m07{yb}h^Lv<D3!A2FnG$O!e"
        "7^3KqINOUOi7Xhh3QG4R=w8+a*2$*CaxzR+hbdy4|Fy@5s;&$(IdP6FDxiXl#62t&pjeR@OdZW?b(eLf<%k4^M3r~E0N1uhU|*TP"
        ";4SeT*C*~+^l-Ywv5CivZg&hV`oFnoj%c<foj#qvJ;#e*%%W-Wbk;H@riRr`Tb$A*Dduwgo@#VF8r)<FWJw;L6xI5x0a@|b19f?O"
        "ktuLkoT>!zL4^#QMw^l9Q{xTV5P0Tfamrp?niHsp7af`Qk8+WtbJE;R{;gef0f3nKSaI;=+zwX2Xh?py?h(4({_R}O{%Oy~U@v4s"
        "Nj#Yf20Qu7rq;xR$(xT59xr^FJHkoF<M76Rc$nG}(_TrJ8(^!Bmp+Wx6wi6IImusIvD(2;u9I%Vw@lo*2L4ooGa*Pc7G86Ziqm%%"
        "2$kbsU^HfL?l>qY?1vfXy{pGR(r~L;8^U^_=D59$WC}SQYO7)NHRqukgx{BW;;}U4_-zXLvJNCvm%f}PM1~B4Lq2AOZe=#ei(fjR"
        "?H>6!B<RWPUVC>=@r??5MeksYt$v%G+bx1MKT+=6uSW<&G;BudP~lf3zN2G3_pbXTg?neTG^%c!TuO7PXNdIMu)q0)pi3ByTrT(o"
        "AIsK}-M;3)+w|M_zU=naZt1#-Lt{({{nis*k*()y3jmSh<1XQ?DC-J&Sf}q!L3c>YQ55Q|7j5wv7kl4f*Q;MgmJ&5yiBsC&E@A87"
        "=mGlR&sbl3E@qQyUC=-XU^>sVUhy=*xHwYouYK2D{Tpz_ffM%zPzXhR#os&uC^Ta7#Jnz(Hok^OY2|+EeD+~>)o$?oanV56&DOUO"
        "+nql5=J9#2dAXAU$Qes4hj!?EwB+21S<<%2r5t{2V+Ra-y322Gy;?Q2ZCz(`Eu`(zW`<mUJZQdd&^aLx>a1+~boPhGo>a)S2il!4"
        "65H<|p54tTe?iY1xbDkR2;??MAll5il_}_Xy6ckek(d@nd!gG->vi9|sc?s{1<xK|bNlIrwsXVyKTt~p1QY-O00;m803iVX7ro5)"
        "2mk=48~^|z0001Ua$_%IX=ZO@VQgn*ZEayMWMOn+FLQTgb8TsKXfAMh%~@M--8K?_H{k!k)xJpUbL6&rLE++TcNa;z1s2^RaodN)"
        "J)kAZhmAz4Bo+HK$ba9NAth0l94B2As2>uGm*H?O-+V(SN%G{Azpz>tyk><~&hWyqH@uc5cT%e_n9`1M_Tq=9TiZx?V5PoSwdSQT"
        "OdIBeb?m^664vJDXXj_*!j`+@z@%lj-Q8BIvMZ#z%h~k-hDuS(TVc2p^^qBnVPxS%35S^m&i)02AGj)OVb9OLyMDI)oHe?Xl`Kdp"
        "%XG!WgR~ALvP#0T4TI(EjWDoCz>#`Tpp-@we%C#!^_6s>Xp$tTt!!Ft9IUA-P-yr*aC?Zqf3jK)A4Uwn+U~YBx`4Y6Lryd;YIJ^9"
        "8Qn0jA}?xgEhyC=)(?dxpe7#HPk#%as2a`P$_hPmCz^)my?X{p&qd3ndHTScwiX-w)K)rej?d)u`-S4IJ!pq=&(BIxF&jqqc-6H3"
        "x?0(jvF+a(IOyv9jKNRKkLZ22V=vPxN8LB9BGrU~wH3rf#iU{u6cptkIS*cXEsndIH@78c)fMvsvKud2R;(5(ecO_hcEk24C>;1M"
        "V}IDud>F`rLil62w1R!l>rOm1Mw>Kw(zP{4ix_T7cD8wv`EBS0xCDAOFT19-z1$5G%63NVxh<sJJ>#_nTUszRgsV1omnNHJ!;-5c"
        "%W|O#U5Yg6T($kY7j1;=jOv3Zw+H^&U;mckfmwbK52d_=fKM~;jDHN_kb=j55o&jBIsub3hI#h_v+21gv?;@trLvBzLhNOEg{~QI"
        "WQE%s@kUyR_h<?-@a_l`)naUj1FvN?e<~(|8rF7VDsfLa7sfu|df)W<UanqL_|U$ZM*44Z-AX=)^2Gdbx#g50<=!_tVkr$m%q4!M"
        "(VZ#*_}aZz^>Gh4yVn&!oaFT$VbvyMy6*3!+w-=qkEzEqaK;ru#fFV0->^SzKpQ|o3pTsex`xw202c>Qyq*sB8B}OM7ey(Zz&RM)"
        "+XDXuPl%j%P8X)Gb`|cyVbX(EZSM~fRBd@7c1qtv8A!%l?iCr+G#r$J0HMkh_J=in=D;qm=czrkV$3z$)ecl2p*T04N*4my1>A0%"
        "+-x9w+Rp9BYV-UBGu@ZF!NTy#TZdSnJBOPvb~nt)M!>-C&!6~HvgpdFP3{2zC7;BoZ>c`{n&k@{t2n?UBy|Gn24U7gsWo{iVokyL"
        "5Op=!9Z+Ztlc9X>?Y*EX3|WBrl=}a!W_UsYV*x20#ScOKBPj`m^SO5#rbMW*pgHf0ZyCr0CES1_V6}r#e$6pt4Dkh&`9wo?M4lm4"
        "1(hEX*D@-9J<zeQpFr&p4}e303CO4pp_s@AjKgo@K@=T;J=!SuLcaqPZ(FKDP(~Fu$4g$8m%u!P6PHFP-UzQg9FBGe4}p##6_^LJ"
        ";0NV^oD3$-hx-riMwPu`^En!!QqlB)NV9Q*&>ZPcn?6#|`~j>#fXo3UPP$-e6pnFDA{jRVMt<%fl*>GcBb3e^%JMA;-D8S?9H_5z"
        "bx*3)R|6IcX*wW;vIOIyqrsq&dQi<BvcD<YjMbacl9{sELc=*+pVIbUlX#@+%o=GM!huMIsn<NJgNL|r7WreEAX#MFEg&UJPW$ZX"
        "Lo0~Za%7#XQ$9aXz;NPl1tLfQWN$qwx*A#4NUG-Eqdo6}pq}NTjdDur)FkPbO_qMS`!P~FOE!xO24xBe{X0RGAmBJ29VDXyaG(<="
        "{YER|ME=Cs2nRu(0qH=d1s$TZ`DOa@$5-~|B4fA1udh@x2+KxM!52Xy=;42R`RNVDLHPLWCOUlG3;%`c3by~b)98x6ZP=piW?cm)"
        "%?KJ4rS);}6#k%R*lyHhoSBmE)Dz<*ncj!1Bg_~#i6=O~6J8iX8)H?c{aKDf6Z5X#a5_yO^>0~13bIBxPDGcuFWnDK8UdqEQJ;`N"
        "fw0&y*wTWR^JFS`dKI#S$Lx6w(_LT|W<o$c>LXtYY1b4$AkfB6Z%AJ99gxEU5n4lm$Y+MGl2>ZG-9o&*MWOFrsf&vX2)X_%8C3US"
        "_o#yZoR^MEUsHw(62KNx9cT#jx|A`4C6{3^J+PKa#6giJx-Ky~1;bje#QpdYN|z^S?U8=@>g(%J<cJT%qA|2kd~{on4mfDjX!WCk"
        "=2OXu2lvs569S>9aTMMY$X!Ot3Z6Et3l<-QQ%~PF-TAe^&KS(_q~~yH3zbJc?2@>Z-ja+hBp$c4nf63y4-^oRo$_&56y52M><0t%"
        "mmz%*M2Pb23#O1Ev3r`ex--CFh_wQ1jM)nD7XTB}lZ)P8r_%?*<W3;x9kslqEo;jVLm-#(<~6KN{f7-imJ(~PU-v!P%5J*D9$lVJ"
        "v-~9<c!SL{a4tyB-M~yCSTVGA6YyJWV3uWQ=w5<?WVE?<zeZHF-8XuDwJl{azge`_^VFyuewc$OCeyihvRp&#rzZ`D@d#{C@aAfX"
        ">XGK&`q=i+V(2({na(^I5^K$E(D#&~`T!PYJt<urFk82x(=ti_6W5Xv7sDmD0EZ0#72<?{*bDhVr%oVabU=?9)s#wjFpp7veK!Oz"
        "?p`V%p8fJb%=Y-bvJ8J#tr)@RYaviYpIO;vUC7eDA1wj8BHX{LlK%e5dzLyK7T%s%I5XHH^nEVjn?;iUq@_yz@B{?E_dOl*h&!R^"
        "QMc?fhe{ef(qcP@OQ2?Vi|+8F;BO_deruE(P4SVKG;Yj(2LOT%R*V<OqZcu`vXA0nk?8(S4w9%GdM0Ss4NxC!O?~4-(nmM{nu_p$"
        ")PUcs&DUb|gUt?o>}AL<pPa)x;<BCf=HIQn;HGh=Pn${*t|O>Opwp3WM7bqlF^IUXg-&O+@tOQRU@{c_Z6FH09P+mTGMe?kH+@Qf"
        "-<wo)i^&UdI<Eu%v1NQVpFv@Qf3>;=Cm$|{zhBMalAwCCDtLJ0T0a&aWEN8qq(lsSvvAA&VQINyc&vHf)a;ccS~h)dA%Daw%>IJ0"
        "comtwo=*O{aZ1d1_YV;Pq3}OYO9KQH000080000X0MBF+v$Y2R0M!)$03`qb0CRF<FJftCZ)0I>XJu_|VJ~E1bYU-aZ);_4X?kIF"
        "X>V>WaCxOzOLN?~5x%QR{{ahMI2($RyqnEt>~YmT?3532wro`{vWkTvK@BVtU;xmP;^@Dp8xIoXjHF6M7l}l7qaR;?-4vzt_p=|?"
        "<SW;r5xgd?s#r^`y5Um%n`^?1;nw6tHI1kQ9KF9FmUn&2EYC^LY_&U4vb`sk>1&WID5b|o$48>;m9}KZ%&ryN@%f8U^7yFHsw4gl"
        "92S}zHRy_)@QyU1<tB*16=)$I{Q-|?-<?^_q-m7yxE^(X8LYa9?vIjN(O_WLyz5v|DcOkYsPNm4S2w*9(thMkPTt=2kc*B>`>E&E"
        "BkFK*WydQ&@}3F(_KrbXASv<odiEY&IX<d+L(pMnOD-!_b6ui2erGf8<<@g^D{}IN7^~OEM+AN}m!Jc6+7S9#ewMWO0-k675Pur_"
        ";>8Q{PN1Q0n8yMW`0!c!2X~pY4>ZwG+9jL_NeqO;>*z{hB<syi*}@ehCpY|l)3R<`GpDg8;{_{kZxy|?gO<UU_r?qy?vF+VK0l!I"
        "-(oZmbjnVvSY1XqmLm|eYAn-wS+Y^6tQd%`9Y|IoV}Qrs@Ga3wS%UF1Aq~;&_Vik*>z0?LE9=RRCq+@b@||!`w)YnBxaE)??tBvR"
        "x<DV%=12Zj7@;H)Afg6YgA2qt$m<hGqiY(6eaxDKdxdPc#=1E;)k5>iDt%8R>v-Xfe&F_2=^N5;=qb33FtDq6rz}tU5ivs;_#`=a"
        "O1Na(me<7?tN){8QZ(FHfC+Y7fdQBlC0Ngl&S333ovs|Bu%;QE5*rN`XbxmFhbnp4B8<V^4IONA@-v+yfkQV{*5NgpxST3kbP!Ej"
        "iu%e+i)K`~q&g&-UXkxMBvFb7>nR@1gyH0$tR47Ut(DH`NC8AA#llkbKsV3=2zLvxU_&j?!7;)Z!BnKp5@F<&Pyi5x-v@#L2Z0_d"
        "p3zg96IzzhB6b=UJO|s?SRHU-ir3YIUKQX?XR=ii9To$em>fEFW+nm1n~MzOpuI}8G&3-z2#Mi?vhNVIK2r_-J9H%C4Jf`9k}t_0"
        "c*Idij3QhZn7uGN_VUj^W$tc=3fDQ=YE|s`osTnHVFmr)2fa!P*Cdt0D@Z3yhf`+h#^mK@&iMtlSP4}Il7?{V1omSFb3f-?rc8<R"
        "YjdFoD2&yEv8C_25_@g)j<tp#Vz<$Od@SO`V(}16A(@wf{Ib{;hp0aHhX!n|7h@=2#GDdcIQu@2t$#O;7AJ4kfHP!*AJ6{!QzBIJ"
        "CxqPCIXuWFUXADE)4PiplX*p2(Fquinty?$5Ct^VwCWb@8w>aVm=ObMmll@>J;;jLIk(RCkN3YDPj`;IJAi-y!Xvi=S_0cw!i%qf"
        "St0MqSE%WBXc=?}Typ;M=Rk`WZZ$b+H0R`+ORhl&_yDTxM#1~Db5GKY)I-;ySMVM?!tXR>2QzBPu%YG7$m;_cC$Bf;{D<?izq};a"
        "@`>C$>66^^To%*X0YG6cxeNe*oEL?+F!EOLERI_rL+LiP=*^+C0+tNX&uC-c6D)M7P0ZoH252g%5g3-R7uvW9`rUA1Qy+L9|0t6Y"
        "_1MVAPAEpj1^9#Auswm5r`M`DkHV>_4X9*`w~0?5Aeu3s&XFLL<R`qz59vP{)2MhjrhP~4faDS01@$aMm=jjlCEA7cJOInF*=kPM"
        "iN(t^H&dg%tukH@&Yt6E`b(IXTV~QxQHwfG*B-Vy^6Ni773sw2fs}h!Ci>Cyr~7JPWRU*DqZSR&=8oa^e+Pg9bi;y;yCNCfc*1TE"
        "9>=N|duwIrP7a*C{=Y8g{_eu<so}b4z2n9(7-nGb@*CBv<#bKxV7SJ4A69I5<*$7R%MD}(jN=^lh%Q*)xA$cQx0dmXDw$RBk_jY!"
        "6!~&bG*<ds!u<04V2RXlgG<@|^g<GCIqa1Gt0Ux3X|%8!2xA$<IScs^vpiholeI-K)4}1WmQn}f%Vz-w{)=$}uQ6>6(q-mL@HE9G"
        "-?yYzJlPpF&Y9h8gm{A^(Ea1AXJd{-&w7c2E$c#wJ^O-Oz)ckcQWfrlJh}w6g7$KJ%WvJf#u^J9fsR$J@|^+*FUa3UH={{oRS%lh"
        "2~RuBD~QuY>E<3-c5f5AsMn)m_pJltH4MXMXxpcP9ZR6YY<WIzC|<OBxhAkBWK&L8j<lxdC(HQjA8$h(l~%CE`kwa6r(r^k;fy<_"
        "N39QO^kd-#Szpb7mMFn^ot`dF&Yi<CDbOS^4{Qh<exXoCUt=Nu^M*c)N`PRmB)oSUKm!{JY{#LP6HrnN_ynYj!4yUq0nxcfM!?hU"
        ";T$movd4_?;UdggGH_Pp&7n03@`Zvi(@|=%nLfW7ZHF4`$?q!8YhJ&Cy5Rq2T2To*?wI}y5VmUf52JeYHmvi9m)(R1$3Y{J_bLt^"
        "&xmOZHctmn%y&wGBSi9)Wi6|jWt(sF0ZgOe@Oi%8TQ>5ALd@Q%u-v$o3JXpJjV0?kF{@N-9zFC)=JjI&VT6{wf~wrR3DU;zFaKXB"
        "_;AE|v7`I`rEs9@J#amiftkVxvZZvW(M&jL_q|Ac^=9HzMfb$_Q4c!%FAu5hg|YuKQG1xC;C}&7O9KQH000080000X0Hp>uo9O@m"
        "0NVio04D$d0CRF<FJftCZ)0I>XJu_|VJ~HNVQh6_bZKvHFJE72ZfSI1UoLQYjgQMp12Gha_XG0|hpZH&FCZ?g9;MKZGIb${giLy-"
        "26N*~Dt&wtM+?>wnpMbu$@dE(TIz|(Yl1-sCSZzzlkr>V1H;_(K`VPt&Z$f2vhU3)eh1o<-X#+}m;ro9U|h81$S2y&gb;OAS3L&@"
        "-B_?K1vC0kcBe}0>ne5FP)X(w;jFG8AAEkqcCo_dNv<Alak;)2-&?%H7MmNC^XoM(7Y+U>k&REe%0<Oy2wo=DS^6dBP0H{|9@<A@"
        "Q-&No=VgqP%33Mm3SP#P_(Lkb`@{t6L7S#MlK<hIb{!KJ*K1vU0#Hi>1QY-O00;m803iTe#NZua2LJ%{7XSb#0001Ua$_%IX=ZO@"
        "VQgn*ZEayMWp-g~bzyXAZ*DJTc42H^V{c?-E^v9(So?3|HWL0lK>mk;iXwL5$7bOcJ?C815AD`~U$DK~0}h2iOSDZy5?zvV;^y+d"
        "-;k2^bau0g9@<L~#1@A`&NnmPN6NCSB|UJVKt_jy&!`6jw&gDI-_sx<AYCq~1mg2>Cz(e<(gP7hQU3&zN5aJkf^Mmxp-W}!`s(^>"
        "y=M{v?nOQYkK^Q!E1-|mjTD7nqJV_<k@yXm4i<|B=+6s3k`3SjJsr4Yii?x}NSFXc<h^#gh&DcxdtB|jBSFFCp`wopc<7(*%s0V("
        "A8lM7;4Ycup+rZ&M+}Vx6;T+naCd#RA(HwmL@y6a?LqA^EE3>u2`)!f1NKDVDDDX#BPijn0VMQ5@p~G!73V%|B5y|(Zp^YQ9rRZ4"
        "z|41jA|-lefk8EWl$fZ~0o|QP=Lf}jNc_~pCdPiJv2S5OTbA0||616~@zkEJomqcsuk@VjtDkcVn13)xf?FA!y4P2LKbB5&lBEQ}"
        "JDBm1LO=<E6ouqSm`^sCGt7g|*pUjh)%9RDMv`o|l6bXK%rW(y5RvPGR3d)3*9oSWROf)Zx$UMg{(I|lqJD#(SvT>pIUHVN=d&Yq"
        "akUNZfA1WB__VcrpiXS*Geuu(dye<h_O9Sk>aFMyKjYc}=0qZIhd5<c`uDv(o%TD<bUv~t&al56;i$D&>L9!AHeszD=R2RaJbG-K"
        "s%U%Mm2I8Dc8tGEgy(cHDXU-Yfjt}k;!N!R$X*WSC1(#04=oJWM2oc*v~kXGIbW@YxVHVVKXI1!VxFI|!|sSS#+C%+4<0t9%XA5~"
        "U026%D&ZTLWBX<0lIu?vx_;zQ?+)uL9X~f8gi)}eBDU0cv}f)9_0`gTXD=}f+Zps%_GCP>S6%R!t2A_tQgv%6pOOkUeAVnU?YD#G"
        "cx`n{rnT7Yl4+w-U%gJJ^`?=_V6T#<*j>F*UcK}B)jKb({{7q6ul^jD(Kj$8ArBdn2QuA`Ay`Pz$YY4OA-R}LBBX9C#F)*H_J)%&"
        "a#)U9%n~bv^)lUfbPLXwi-0J{B)W!W5LJ7u4$Yr5CnWSEsER}g7G~H2wgoc@P$mooQD0KnT2Ha7*6sG$%2qoF7AhwrbVq5DO7KL4"
        "97H$rmCP4Oyobg`meTc|S2$Ay_-7^N-zEQ<kD{bBTt}R^(VX9dcLcYJ1|~ec_l#Z_dY?HdG!UnndZt{mkrd4WsswW-Zdo3Dasq-#"
        "wEL|yv972ptlwJ;gbQIu8P$T6^$w2jDXQe2gqq1^B#a&#L|Tb6+ym-I`W)`*DQ>!ysvCxgV~r~pV=Ij?{_wy&tfn6P<Hz{B$xqGd"
        "S@2yvR!=CKY^a9J8c!x4R_kSdJ^p9|a?R3SL2Wi)!_@AtJ}m7~-IPQ=-nb&H)rZAmGPXy_EGxg@)&h7kYdm-1Da-M*RWIxdAk)2r"
        ";EKE4$I}U7?y)Tn+#z;Yho-1<rxmFm;CVx55Q&QmrcSMr>q^Myoju)5^?q);|9t!7GyDghZqic^)U?|Dd|TIpZ{k6H4$42j{a!zx"
        "3<q%h0czjYOUpL%%f=(RGIa0_L#{}>aDhczsD**7)w>xL#U8Wo#+UVEdgZ+LVz+Dmt&+so&!JcqPZuU<l3zcEddaL@hV|#1_cZWJ"
        "kX#B3x00iQ%c14Cq=iSj+D#hfP5n}cmGCMKbf{)-9Of6zEys_3&Ir7+3gh#82PWugh@J-8lP;;0qp)!*!eaWT$i4AN{Qro8&BdW|"
        "aUuDui(!uBqTo{_i5?@nkW0nfjE7U|)STHZ9V}ypubdwZwc^s;KWfg!_2k`!v!hnO-0Dc|F6shN?!I9th@^o;c-WGL>gnH~jG=~y"
        "$o_Z<Z5Ym{i;2A+&u1_i_utRvtMzzT|C_OOp(iW6jYsOH$u4#gCLGB)$J^Q9M=Dx{!J*EyZaTyH_SDe$rSARf$-G$8U&iIt;!o#a"
        "GB4EsUqI-^P`{Sdt3iLQrve^dPVia%<r)3bEwI#PTdP=~7fBFZRWFAxs{?)1ofqKYwxAA4yw2w2n0)0WqAV7_{VDXXZ#7%?sNY*4"
        ";LRgS770~bJ@96F^E@-Fio8=w_NMnGMA&oMhnrrf#v`e_uyRRi!AE^ziKi56tMSsyh4_qJ6oOh=2AW0&00DIoPqL%+g!lV}+e3W+"
        "&n^*ogSabPKv0J?J<4c<3009Yn27T|LFY=A1tkjaL;estiR&I}rmB^bh}fIw6-1IMLjfuJ$+46m4AC;c*+OuE#h3ukBV}~jyp3az"
        "O*1vl5!RwKI$SIIt*3?kKNh&S32)zYN_~{OgS?x6Sc2x~CUzg9p%rkvZjS8ui`K*y^k*Yj|2oG3NU6Lcfm5A*%vC0U@rlB`#`F>i"
        "6!A;j%KKGQZUGVZvLt|CEp$an`}0&UL(>#x%H|h)EmuM3yK}EXE{0VzM-6y-u76e^KPw~2dasv|i{$@;mrqN`=RLh7Auq^CjQM{7"
        "P)h>@6aWAK2mk;8ApraYW(lqV008;|001Tc0047xV=rQ9W^ZF*Y-eR{ZDB8Ec42IFVRUJ4ZZBwIa&BdFb1rasZBk2X8!-^R8~h)J"
        "JvHEN((4`yTVj(yNJ5;pM>CdYZDBpkjFdipeMh@>oYLqZ7)kTJMorT!uRe8vCKvR;bVSFnW3(>rz{N~FprIDlL|~JV%jLKJ?rZ;i"
        "`*7a|+cr%zpUr0jrvOLHd+)a3!ZC4%M=f9Pwr)rU{f<Jw)})wO8RhhJ`GUS+b}2qlJ$SE43=Tu`_SCZ?%!TPb<tIY>Yss9r(#>ba"
        "qX?|OFTYSU`y1p)BHerjx-1Pq6B~V)j_MNf!vc;>)|o0pHq}fwdRTzPj6!+C^x{NsJCMu^;1V~^j>Uv@0UtmA=cPNdXjrRKMZ<M6"
        "2jYqli91;Jw|AT8hs|TZUN4~TrwTBt<2W!XOEF_JRjblgt0&~^N#UgeLcr(-5=}Q0$vRS#*j9&RGFi}v2N_|vHJ*Tc&{SdyE@DpH"
        "o<VzUVu#-SR;h_Kp86jwm=uo4PEu5`ttW5-F=arIVvJ@z6n_d?V`%>g;9M{|p?AmlW-w!y{?KLt&EL@W(?EfKk|~v7G{YptXVt->"
        "D#qzT)2hmTIGt?!hWlbZ`vXu*0|XQR000O8001EX=IZSZqzeE5a3uf$EC2uib8=%ZVrgb?V_|G(Wo>O?FJ*RNY;|FDX>V>XZeet3"
        "c4c31d30!RZZ2?nwHohp+qU(4?(#pN?1z%8%s5-S+o@GL*OqJDV##Nz-EwY@2a}M+GexohXj}F2zwbQ&BtZ&Pl5SHj5(zxK_wat>"
        "!Jg;MsZ1Xjx!TE1k&|1N$D1wX_e8`6PxHGGVNYe2#;GK6kuZ2;aV66tACYzX#1fLQyx68Wl?4~!%NH+SoDfmnmb{3W5U(Cs{y<Ve"
        "@<I|y1gzz$+z~<Z<hFPU$%+Xn2(NNd<k^lqZdgt@`>ldumH;}>^AKmcEej49glwJ*(L8e2Jhr5pA<4GoI?Y;`s!EfWFV?)+l7vbc"
        "XH*C#NJAq@V>xQ43}{K^CQEN?uq${)&eYs(x-Mdx-LdVKhH;TgPGi|%yk=z~QjpJXnbT5i3h7#g^we7RB<qzi&5Jw*kVu%7(T0l6"
        ";Bw$G-dI~H%WM~^!AKSj4pgb)Fskx&ow6iK()D@-N?Egr1GBh9ZSLiZzc*s=f#&ZlpR9Prf|oDU7}3VFqT>FVi7J!hmoEr>(p*TI"
        "$1F;dF%gpMQ5chzum&_|$R#i_CbvbA>4|lf$7o})@e;`BDe`YJFLI_KIHRJ-6)rL%Xe63KQdH70$~d=2@n6E$%Enn#D%00;^4%L4"
        "8B9ikV~0)+NXZ%6PimMmqY~^n5;0FpX%GLYUZgqDzC~{{W&?r$u^7CewE?lAfByEbHVn<O;xR%8jT#3vQk#%-nh91jl5QD@C<0F`"
        "@&qKi&I&34=r6~8a)zk~#znmwqXGcOA6O8Oqc?*_u<lbM-%rS~nN>=psbJ(6npJGddBJ_JUa+k|Ubmnc6fz+9f(GY+Px$p)K)#p^"
        "%j^(w_5d#wMb||gZBwyD<EXiid}krjVQ@0!ZT?}m^IL-nf?5MHD<$z;-bQ4xG!Jcn3s+iL)2A{8905cw!Vt*bB$E}-_cY{N_B|rr"
        ")yeAgB04{ry`7$UBV#Z152)Q&XoH*ax5vltTTQ5#?lQcO({CYq-Ui$(S0}64X|$NXeFw~7xEX14BXzE7D3K%;=q^d<wHUh$TAmle"
        "R?^ZBK3hJs`T6xpG`+rFT<>vI)R_|r5ksgplrPCoMV9C|HzJ!ffn=!w1%U)eVo)2)WeS2OIRi%{@CM&<!^jpynt^1>N`@NQV=6aI"
        "G7DD>V?5=%Gf;kvfe0-LY;oe5yl{l}eB>#+$my2Z+b#<9dCXxxP6Z)nqR1YYA6Rx>aDt?O32oyczKL;HcbF;wm9J2~>g5n$v5HTD"
        "lZ2%w1PW6DYcr(`a3O^==+XPA>ms6viI|p6)UbI4Za7Fe+}<ZC_w|dIXq~dB6pW>~uXEhq+D9l}SaiDs73!_2d=1#3Y65-vUM1H@"
        "fAa!stw06m4CAo(<LrEKdh#}!{sLd;v-xa!F+Gbs^0g{SLj3y155f(5veqk$g4IUAz84w&MrP9mccOqHFD!Sq48A1)V60T;wT6l<"
        "BUtExy;V7+(|CiPMCckVkmiI@4h0#AM4|y`bHq)CHa>`C674zM(lqxGc%s5afHmJEhmZ)NQ50lN(+rBqS75YQ{Z$}F0g)xmGBsu4"
        "?8a6-!Z71N?T5M(%(q<H+MJ1+%Uv$%lQIieKs@C1`-b+(VPKYnF4BZFDj{{UN7b|;v`fNztpypOI||tDvsVv!A6hUJa|0rn?E7nu"
        "B|4}Jh4o0Ydmn|@cOme&RMcDZamh$6n5nAK<#1i@!m5O-kU8X~utSInsX#~q9I03JvNmtX9}*4qK9HLRW&&Q)?0X;oe&{%`k#ne#"
        "0~Kc#HL&0+>rrrQwUzMyd7DVcy~TD=7ui1>b}!M#aDQ_^yW%t#Kx@l*Ut?I=K7e3kbPDO&7c7I~9$9gz7H7jdmNTBlky@n$891Cl"
        "M94D+`vH-3brE6}uN874A@kl=(uEd3jgoLlNF^5P!y>9zxafuGfW{2wnAtvspjs1x`@-(JrtZ)WYTF;)sv_FMuQ9il<r40?majuc"
        "_)*yI+3vC62|EAHy|T3X#>|>IKy@Edeus6ihauT}4>9P*jaCT**fy(uSU1s&Mns8ag0rWE+O70;a32urd-t#jw}>Y$%3YK}^1@6#"
        "U_~`Y%65lA@Popg)I#^uWYXqFI0nb}kxDeMlUP0}xVHsO{2&+|Mhh|!lN%K3`;S1@GN0TWf;>b|1ugfqL&`>!xX2C@vWW$WNt9Kt"
        "Qzbq_?S>~9u-Lms_QLN3l_1ofud;=$-VNnB95wcR-mCj9JMnAZN7o*VhPt531>@2`9+78Ef~uhVf`171)_54ST-t-Gwz%VT8;tkY"
        "Xpvy<1_ljWs0ZA@;2H)V$`f^x1sBpIp(@L-s~jWGF;Tfx)^m?kEUOp~UF$0IhgebUsM2(~6A<gC;70yy$6hT}8m{+Ypuj0gtb6XV"
        ";ugHl-^a7|@F3*J#kPc6)PJ_j0AxHOUwyAn97j!4??`R(%~vpoOTP+6MALC0FL??c|6CWadup?bTz%}(<uNv}d}BCJQ)6WerZ=f;"
        "W4nC3O!IULbtuapQeNa+AoU%{om`y1`YDwc)$Oa3t6ArmgvmAEFeH?at9Pr5#T-tr7ghnOPNd2ap7CIN;o*_;-EuX(yk0C;`1a4)"
        "{A`9LoOimoTwYC2J$(kgnEp%6zdeiI&VIZ;xqcTZ1Xlfp-Aq6Xem~aXQ3qkRGtm;GaUnvppAa)XXikM>wu6GAWE@}fn#(R~cg1nB"
        "xSUQJ@971LpO@3?tLw!%9Nh<DR+m@9bJIhUmscRD!D%i2`2u-Z&Q{Zj_u6xUT%0T~mec9k#5;Ba%ofx6>EdiQ|EV^Uk<-pVjzjIK"
        "F4`Ij&rK{8aF&TTKKFi2zWz0YFaL+J`VD^YMnf3)Yt06!2_CMK=_em86wc2+Uep|zo`YdE{CKInf2_ifS&wIlwynggq11e!hS=+1"
        "9>{s#(X4L<-Rbz?w8=bWMkNNj?{u81U&F2zA=?ij8%o*Wh!glfFdNenRLr8Hl4T`z<G>Axaq!^0Zr@E_A3FeozCG)f;2&%Q)}Q~j"
        "-8Ik)tUolC6~#_RZI`+qU-#Yi9I6J>Y^m!dlXo2kSR-$nyWt%#D!_}2^ej-KTng-z`Fl{<59xP=oZY2A?89z|t~}i^2h7BD+Fes?"
        "&%^<6S+pinH>$^Nq<HSj0SW^n%4hxiA#`zN32HxfG)31(!^X{iV9@@rY{@inT%H&|CKF6Lj;Ub>{WRx1<e2?(#RY;6ASNDX5OZSY"
        "0;yfQcmL6Y+;3hDdi5uEu)2P{nq5v8Kd%}eHVyr{i>?~hhQp3y7-meY`P?qr{~(s&ouA3ws{FR%gDTGPblvn-)4T@L(Z;Gu)qNv2"
        "UawA;OAjNC!OTDq#KQ{@D|5oMcd430^Fp2%Ri3E*dtC~^GR7NE_m~DCdZKC>vBlA$th&2jxLfjl!SCOi)pr4r)dxq6aFC+;9ZPKO"
        "e*Tq=Ycu^yao?M8s4sF~ow;DJmPk(|8cH`Adg#2qpLLWT8fWxtvjEfkehsG!v0=#@FSt4S=J@z{+z&Jzr8!^%>#Dfg!U-z<okh96"
        "pU`WCZ+^!lONG4Yi}k&_A;k;6_Et1wx&N%1ftHjzeM5uugSyK^a5E4byuVB;{6h^mK-gwrkr1^RP>B|!%VQh6o^8Ajwrr-t-i16?"
        "z)HXC=KO52ZSbhe^^fDhbg#_f5H4f1&+YcTW)t4H-H@d_&!VMS0YLp=L+RbZlztdMfRhmi(KdkOl_f?{YWt$1liB;k8dtX6%jO>B"
        "1;O}(HmN^I#ff-bWmyvsx<Vpv>KMAuWmh!ypJ(i9u9Pan&fjflz1b?WneGHR$EcHc9UYheg3qqtQ}fYA5%aXiO0TDZ_c9t1U<AZ("
        "miW&Np8o0&*76TFe`X%oyU~9wWXwBo+i)FG;IRK6P)h>@6aWAK2mk;8Apo`$?x@!R000&Q001fg0047xV=rQ9W^ZF*Y-eR{ZDB8E"
        "c42IFVRUJ4ZZB|hWn^h%bZKvHb1raseNxSC+b|5i8}K_2I@wra3_A@TveQyv!_XD$upPR>kR>{0D@z6?F9o{3`z$+coGdL*BKiGL"
        "5`_@!x9>`DO+$*Lfy-a(WsO$dz?aK!=U-t<jL<l4W%vLV{~!ZdWqOUUaYlpWK#f$v7ea_MNfT;2#{q7=vuV<>Ye6|<Pz7?<7qV8c"
        "LwpX%vNpIwbu!!-s-L<SLmK7AUQLZt(rmD8Wucr6937va-?5X#7mS)lmq<$!jSXy>0#<&D<Bh8B8RE0qxqKew>R9S}G0s(j4-9|D"
        "p>kM$gmY)HOcNM_-PXejuCAxlBibSL7_xk-437tUx8e0*?mb~*P%~>3WPOlUVMRJpqU;Pi8cXsBKc(q$&Fna9#HYS9q+~#t(MxzZ"
        "t@|kQy<#*dq}JJK><k$%T*$6NtFyz2;-43~Z##d~i1F}$DTbJti1LLOSiHb(okmkx*<$6mia&KIvG2dGFfi2-bu67=J)Wqc^*v)H"
        "y&_tjN#l?0X0Nw92FV)?!=l8M%Hre)jlrJLhN8XG#MzYi)er7*0eDY7RPJs}4_VW`2fPoN_-#dT>&RyDzjAm3V<~RgxUo1)2}0z9"
        "o>fkh-K<0(n&tbr6#oEFO9KQH000080000X02JkHTtEN-07C!(044wc0CRF<FJftCZ)0I>XJu_|VJ~TJW@U0^ZewLHUtei%X>?y-"
        "E^v8MQd05^(M>E*EJ{@<D9TSMO)kmI&r`_EOG_<E%}Y*IC`!#u%*-oRNCXL%q^1`of)ym?<R@qADJd!OasdEPO9KQH000080000X"
        "04kA3%HRh80OT7004M+e0CRF<FJftCZ)0I>XJu_|VJ~TJW@U0^ZewLHXJu|>a$$67Z*DGddF@%<kJ~m9e_vq#2Ubr~st9(`7FXvK"
        "+)G*%2Ly+-*OzS|&=O@6E0I2u+U)k)|GhK(kSJMpHpN|m1I~F^n;-K%znNi@B>8@`CHg3pAqT}+^F$PDnPQ@5<iG?|)Nm<Y5h00T"
        "{Yfg?k-z^$YFZx|k&3kRo)KM3#dMJ*$?eVUO{-*28fs|WQLULK9PgQF2@m--BUODIFYS3NYuX)H-_xR&qUDEh`OV{rDc&<-emJrE"
        "!g-@;ZAO0Y*okuW=8^WNj^+O6KNM?tjXJH-KoJWg@0sAgSj7XaR;r$w3d|xKrYhvc9!*X<Nt@~cBx^ZqDu0uyHQKtSZ7aJbko=ZW"
        "Gbr{}ns<ygPj8fx3N3Ma^S9AbDUkVviSkD^u<Z86u95GopS~NNv%9?^@UhG*-sl}M189851qfIa#e27}jXYI~Q4KfXhjObyBYPXJ"
        "!X>xSFt>4nc;Juoef-z3)<;a`j;8Kj4_~`{PJB%$f5el&a^i*m+tbHNun(~$fsNFw1V~BJZBDkEuXFO%CX478X=qmk6(fpp<;%@C"
        "U&GOiynKDh)DY-ztZGDJo>0+1*3_~;4TeE#3>uQe6DJV1U~P7KMfAW8-FFOwi9p7{6(blp^u#-9eEJ}H-V$%jbG^x5$#&WULPDCz"
        "h_kJZiFDe^$mdI1uP++I(aClbJw>@@<R{t<%!X>3Oz=?chvvW_JsxYY7i*Ht-FtxO-wz#CPheLY7=>r!v%9b-Dh)X^s@`*vrZ1w<"
        "$;pdIiwrEVLG3{F>*rpVkpNdU8S@rzR*kH{s8!2NWo51=DnT(6)2idy6RycyDO@PTQGbo=o@&-{!K#`HWC-!V_yeovDH&Kc?14k9"
        "l|%sWq>lTED%vx{luom(a3@9q-Os{2o#0kU5;#bc=Q>oee+?Wky#mw&bfX<ur>Q<LemI&@B(U}amxH!u8a)gk%mo*R2o4JNK)cl6"
        "l7oRff#Y``m>ed-LIMGMFvqInJvTlV%*=HkxH-C-QE=hQl;xhrq3WRrD7vEt%n173MUHu&lLjqQx)YrghumANbbqoldSDBEKN1CR"
        "YmoGwrLZ-pwwJP7T-OHds7D()NkSQM6at5AHM%g(ZG^E7yoTTA3@%0F!Ym6_&gp1yye<AE>9{sAe#{Rdh*bcY5J%#`(mAp$y<sJ6"
        "^qEHjw@-G54Xt%|SS0v}JMiO>2nYajQDc<LshlKkqryC^I(gs_@c9hA{OSr#w*fSLNwl({oVkyH$jx`VoV?t=Uoli`y^Uh0KGt;$"
        "DP>sd85gvJbXqeqe$EqHAlM>-DoSy$+KLv|hoA)1L(j3G>VSCra+?)ErU&Nd*IZ&3^mJmPNiC6;Q=pxr51|tZ_XRyhTrgmYYA*-T"
        "AO=q)0M@?RxQs}ER<cfK4}|W|w?gc#2@KW(6UWFCVOu)Gy)1wjG;`vtVMuLDk~-65xz{PTr8$D5EMFGuZ0fR<ZCs%dg9gLkrf&{3"
        "G{O*1p}(Dqm~GCX8s@|UjP3WxuMR+5*5|hL&r3~-tq*iQQ<w9WL2rvfL+%=g0jTjOYU<-F@E(GAPoJRQQBdD0*b~c<VIBM!8i*2X"
        "ij3lj{Xvka$MJnY4j58i>Y`KBuozD+6dVmNU>BH`7n=0cTgc+7UmNSSjUB&u*0W;&I2C*=jNgp{!>>F;k@4iZBInFq?D#c*Jq`Ni"
        "gf>lugrPbI2iiFyq6cCxBh5E`i@txJ9_I)Wj(@!8!;cG9G~b+V^3G+66c{9A5Y|ql;cZLWp=)^u{m_9J7%jPgkR9xxMg|>LLvah?"
        "^s`9L%3$Q;P&5~%+~UO*QOuFT#K#|@V4EY5;aU&uxl&-8?IW(R7kyX()7^i|LLkAPU<u|7v1En{hoJO58!`<wP$9l{PgM=1Dttod"
        "03#x;@BW=A*|B6tl0h>Cr8cp03j<5eOuvOu&=U8_Q+`W9`e*~wAqr1-)n-H<Cm2p(6}VO0gCS=`rmhw=9@8&q;nUBJ{VS<wDP<E}"
        "Omi6yEAHFf<~{jrGRz*zUzpmTJ!!$6lk9js&LY0@%myLHE)SfdiQQ3R4#uNGvRk~cgwfhX@^0~%^0A40oY}EzmbTtRj}#O1#amZu"
        ")x>V)xndgG!=Y=_{a?1<d=aMm<%`g6KHQB(0{pj-d!@T}UR!(AS*G~+n=MB94+G`}WE>U^NH_rweqf$8&XJ*xOw8T0TM;MUVje|T"
        "Vu)FnJ~O1;kc&5}p6eb1BgB>+OcuceM!eLKgkXvjZWb{sArS5O2q?qr+WZY<(W{&<&C@tUIyMGt?^I)W&d?zi-RphnC8nJJT%22r"
        "!Sd3^Tyj_ZNBN8KVQ=&`p1=r!zjrV~U@ILzbVZJLwD8<+oHcCfNBA7iD*B$E){$)nHp|RO$BG9<iQ-Vsp+T`-kDxs?o0ey`f<q0b"
        "2O+iLb!D5isI@(RPmD_595mL^<>~`sx!rv6<wayEp8+Xlz>?j6xTSS(7k_^1d8y?bU%Nl4<NSl2W)u*1#;7DH)pqj3YJ(ZKiJz0?"
        "XYo%Uhpmc6bVcx7HfzcCe(a1O^g8fmHX{7LzxiJW*GSA2oQ<&3hxE!J3&3laR{%LO0(RUNp?warpN7wQL^k@bp4*$>0Z>Z=1QY-O"
        "00;m803iT%;)EM10RRBx0ssIb0001Ua$_%IX=ZO@VQgn*ZEayMZEs{{Y;!MPUukY>bYEXCaCv=^O>V;=5QX=ad<Voj$_=_mvdF4Z"
        "m3CKE2*bpt0tOA^roMf#Q#<`p%m!)R<73{8lydLRXtX~<2wG?|ontWEllCH@?LFj;Fg+1O_Jd%^3|3qmcOnmwJ78x@HD;u|l~UGK"
        "U7ZPoXs+CL<ZcTIeBL(XYzxALgv8>g3+h0a^NtIF9mf4+SRt5lsg`u=R3dyYog&S=-Sf+FRGwyGg#)1hYh7&f{932v2X#TRY7tdJ"
        "@652m2|z7arRM-M?F45iBQx+g8J2)6sHoX}EW>Gf=dHTd6uUU6$+q&IurI#$)<MDpMuty56$NM`rrVCfUbE@ORVk&tS4upH*SZqr"
        "$U7g)&9{DcYm4Jv_QkL=>c#VKnLqG<g58wg`NsNNU3~#iO9KQH000080000X063DA*mDN}06rH003iSX0CRF<FJftCZ)0I>XJu_|"
        "VJ~fOWMyn~FJf|OWM^eAaCxm*TW{Mo6n+no|3G9fQY$N$wR`a30lj4{x^-QeVFQXnpe4%YtXo9MN&2wgzVDEdOwn@IZZm?`A`icF"
        "crM>L^2y1`FI6MUf@e&Lir2gm$&5FRC}@?{A{DhLV!_g~W_+1(T2=uo%7!&-!4i@3HftEK<F#x=+_bd_kB^Rzu4z-&Sebpeyk-rr"
        "SE6C95{l&l-xIdelGbI;WFZ^LGx=GoV4e~HC}VZ{?CQnqH`lLTT>p0UD$EmK!@udJfwy7{fbnAOorPyl{eY<$JH<*H7}upzqZ=ui"
        "w7Bl_o{@NHOqE5|HoHx!SX;zvUck`rfrIC`6lp4B2^`uc2XI!2I>rKD309`8!g*0|MRI(Uh`3Bd&6Z-#w-Ua?AN;7-Ry9uuMuqH3"
        "K-rHX&p9k;1a`EM9lp$~OyrOh3Cuh>IiZ_lv8)@G^JaZ~q&?T%S9IHMqmQR0+mwjT_$e$3#-XLyTMkp)yh{m4ToTUlQJit5*wxP;"
        "-j#LkLF(r+X*1!^kB=BWunl`Ky3ThYM6eTbvv%LImdQ$d-}!ATYMDxGo4aTH9)77smGM|#IP5M<*hn!JMU);<gsjNY0Ev~)$zcH#"
        "m723OD|v$?I2W_;dW-0zTExfm!{M}V?ExsnX&l49&ibv8t2H6wqVG`@xo7}nJX97!c`(5DZ39srx4cd|`|jqQJFi8d%GzhMA4Ut7"
        "jayzzP6f7LcVSiDdKZ3}3to7$^MGxHsGw?bUAMxry%s35g59x49cry^>c$(O9uK$kA!6Ogj^&S3>My7|p7q7?T*qiKx>}lP@!25H"
        "%mI`}QJ~N%37<<-^d^5^bs5hKoaL>;(o0hqVeh<W^-5Xp=;|iY5q>R`L=+Tlc#ZU-#&k^^sJ_NL5w60ZOQk35)syGM5j|p8bq^y}"
        "by<~^u6>QVm=`0=qRpeV;0bc?U7If<51bvx=q*c}o{j#OgJNW+UJu!ul)Xbz1wEH41l9HfJ3ZYgv``bWxozfb7msd<uCwGznaSqf"
        "I?yp}v_dSSEzjnxV^K!r$!M9SJVr`|2$SJ;R4%b3^DKg8f9bWevnPJAL)XP)5if_QlO3Z{3h*o|Z)tyEKq}%AoKSogb=h$QS^uX="
        "A^W=Yxh5t_N@3`6)uJ>(j?l$Yc?WF==y=4!JNDgSgQ=pMO+@rHXTPrn?5`;bysEPMp7j_U{HwmlTL7ZqsXsL{SevwQ6Aq;^$y;NB"
        "2@WJYQQCpb9dB6)hz+8??}Dj(>zlzxNPU8sPPB;>>KPPq4IWAuOYUiKz<<lK%pgLX)rn~SBh_ES&2|MWY&W?aUf6QhlOs2{jA*_8"
        "?#T3xgUIkzrq*|pcytveG9?a{se>!QkG+MCF^|TYor6?R)EdsiGj^(beyFK_bjJ0Z6%8D<p8CwB&o}J-uFM>On}|x#w7;KZA$aXs"
        "7Y&>{1#Bs~T3m8C#5HBJk7PyYYmm188xj(JAIm*Hy6*_fcHysYXYaIMhl)w}LNIqI(m1#aPCrl`il^nsFFZkM`JhVN`AtIoqPh1T"
        ")Z_>{UPQMmqQ2i~;g{yY>lPhX(E!1!ewLc=#(mJUYU4F}2AD-R8KN5>m@fU;aHbvI`I?KKqKdM@^Ug8(VCj_M+YI0QF_lHDL0VT*"
        "R<3XcmGyW}l>x!lY=6tmC*3zEN5w`Dd)`N*{1dVYqq<m_O--fWUQCaX|JW{61FdR2W=^z@FBHqU+E~rPmUg<vL!&Hu-=(o1fyRVM"
        "uZ-^Nmuc~#og2;sICV>-ih8<Q^GYyp$(wkst2D|)F)CNznHgMH9KwQ}!Ln%O5Z2gVy5nJFr|AMy57K2T7>VpFIoE%kDyRDB0|dt&"
        "pWR{@5xSk8o+BF*8Pi4Z;*iM%q1XsLChf4U^xJ{!QZvA}RigjzN9W!tU*}dm<n#|qOFptXwSf*b@=iLk935HxdMv$@zv(-s=}F!g"
        "-SXh37c?!$8v|Q{+g@;cK+vX{D`7l?S+IA`(7LJT+?bn8@Gu#E6K<ur^=uD~yMHil{(jhCpjGrH;vr2goa=1{xWR@07LJ^-{qJzL"
        "tDfmu`CoKIE5x+x-8gk&MON=`T*U*qs2D@JHO%N9Wfp;9SPRUbx0!c?muu@j^s+qz7x&|sp1W){xQ1Q)_9W<B$B_)-IS*}W!7k?e"
        "_5Ka}FhxR|Wxtd&Ms_Vmk00kj_ikzaoqzfVTd6;ht;ScxYZFF~-D*9B1R|0sm09NfWg`dX&A~gF$s#@Rom))IED!2I!?wZgIB6;K"
        "V+S3&3{{ynIluF)<NEib#q8Wacx-RN@oWJn|9;o2dW^x_&jd{j#Xmt48rO_aqx`h>1#EkOmX<u;*xWQ|8YbCkRsC1DqQkgEC9T@B"
        "Rd_X<Ca)>z?u{LxH~6DD+v~MWOoY}C1pSXlWie-;f$;~TY>k7DSpQp9;r$V?*OqoYndzTr2QTV_!*mAbDxJ<@Qhon7wXqaTu)rUU"
        "?|FRmH&9Ch1QY-O00;m803iSgX`>3i0RR9)0{{RZ0001Ua$_%IX=ZO@VQgn*ZEayMZEs{{Y;!MTVPj}zE^v8`Q%i5#Fbuv2$bTSw"
        "YJtYtX$K#M4MPhILwCr6T@6K|V<ECAkW4!7*N={yr7JpMx(F5@--C2jRgdr9H&B?u(QBj(-uWKo4nh=-*(-ER189#xDL_o=D9TRb"
        "2|+M*tFJ!*Epu>6Tvt`KTr3yolHk<&<U~*)4ux2+PgBzKo0a+gPth)R<Xzh1Ft}R0>$)@Qoce%HkBr|)H|WB6&rCd3E*BqPbUt5j"
        "pFIWSM;wU9X1M@6p0eUkZM&HF$c1Cp229Ny#krz2`TkJIf-U1Iikb2vE8zcR^)Up@t*BF!X4Y75G0TPUOmc&TBwBR*Jh|RnAUfEi"
        "w#4ns>1Hze?w|B;VfEonz0<LYJRB$2Am$F)8rmRp8XGv^Xf033HXe~9t|hx7J?UEMfI0p`UO!>n_bTg8-?vaGML*UP`%Ci;u4abk"
        "{lKizaGq&`>l7I<M3ox0>|!?F^Ll~vOs1cTEzSgVZ7GUHz~x{pan(~X-Y1u$4S3p4X8@Vhu-q<TfZb@&QO=<3yw-myzkV3J;F}9{"
        "GyMV89aLY2l5j8)uMmyO<o|l6Z0k->*-YzWsjJ)N;x|xB0|XQR000O8001EX$VPJ(RRjP4r3(N6E&u=kb8=%ZVrgb?V_|G(Wo>O?"
        "FKusRWo&aVV`yb#YjAIAZggLCa$#<BW@U0NaCwbZ-;d)o5Pq-J|6wpM3D*tqz)M7Vfde4{TIsa+(p^PvGHFc{JF?yGEldA*W*nzY"
        "+8%p;i83C~&u>0k2=T#NReJFF$=?~>ynvH6deqKX2i{8UYP3)r3x12xStOa#Sb?<VR(UM_-m=`RF`LikvyZY-mGtO9HVy3X#eM6x"
        "vZMbsk`)B0U!XMIO9p=#(CD9y-9bsp4wNP*%0X+YeZ%I25NuPmov|J)&S$kXEp*av8?}Zgzo*}knl}_#)Ep-hw^>w}4mBT{&nm1>"
        "X%<|CqTFJ+>x|Mqna_aMT9&)D(YOeX^Wbj*-z>oGR98^^mCIpmOan{!RW^<lE!K_NY<<Bso$k(ODct=86_q{j9k90gMEim=$0lo}"
        "-t6TDi~d*iT3*@>deR!=$(*M`7i=g=dma4LQAemdOL*-iyr>H~9AvY{vy3}R@<lHXr4RNq>S^l+d%RQ)+IUFL*-{THJ`76=8QV2h"
        "g*Q7YbM}m!w1}sV{pVTv?YZoU^ETl^rVFU8Y>@%56rC3fsPLdlT#B;Wixi}zmdgrlehx++q-x~4LCVmfPkM!nah9Z`pj+jf(i_rR"
        "#p=Eb5y!MS-&3t9!HEuMOZz8RPm@|i%cGch(NH*qk6g?5)*74C;?Eeu8BY+Qn8RzdvdVvWI*RoF+A@SWH0Hu_7%oaP7J@ohxQ2?D"
        "j8jT`IjfF48#D6zuE8YirtsA={)YgvUV7p4X2RDhAs_J}1W5U|(wEzi;46i6%)yzxEfHRWkK<wh`0KU*qqw$yjnL`P1&|WiL_^9I"
        "EqWRJU)MHB7(s8fRs61%SHyAuAfeTHrKML!gCo|Vi5*C8-1A5-*|^PWqe-peme??@s}aa}(7JwxPfffzW=GC^G?qy05~kt;P<^3`"
        "i2D<+^D88970=4dFV;IA--xCZuB^piZy=;Jn#wcg9A1GoMmAOrQNmz7f4U0kVpzY}Pmea9i>%%^P58ZmQ7anxg4R7|$z$~{easv-"
        "bs@diB-s^oul@b+q5td(>RC&F{2|TQgawB<HGDB*y$6bg&`p+Ya=uWacQP_zubBIdctal7V#YdOSYdnh{Zk4*0vEP7pWZ&uZ>}ci"
        "p3Vi}k=Y$0M2nz&k>-tNOo5$A!iKIArl6@VL-k8or4|pmm0=wCa*hm|_Y@-D;(!Ye;UWspgG`%ZV`X(kr@T%aJRb13nRr-{)+UtP"
        "hdt%w+o!*VUz8;`@bmq{T}!Elx56k8eml_&8^!l9O*_G#TS?RjzuiB;{U7gRk+RzkMz%7xRV01DF&=v0?X4ca9z=K*^`VPAUX_c<"
        "ez?^NvZk;imG6ctb;QPjn;4Qq71LWL4)OKKRz9mdwH!k_3&yIbOL{`to*#?IRdnJ^nq%!Lcp3aLE<QnTN34>m$|NjJJj9h6a*3*Z"
        "cHbrp&d+e_=uL9s=D{*Oq5cC<O9KQH000080000X0Qo1z8*l>v0D}nt051Rl0CRF<FJftCZ)0I>XJu_|VJ~fOWMyn~FJo_HWnX7<"
        "VQ^?)aB^>IWn*-2axQRrl~rwT(=ZVJK575qR3wy$G;}Z~R8%AwFpyv%*f)eCH@UWk#16It>%Zsx(!>b`M7|`p@1A>nch6-un?3t<"
        "19#z4RtTSr>Ppiogbzx7#nNyo;GL|x8et<9yp>`*zm&q@t??e0vS}6WP-rsEPft!yf~N-EwQbF@0vkYBW3z7&-n`JjL<M>;47-I2"
        "TeAZ$jHF$_W{p+F#TInSN+iG=g*eyT7*s$}6?3FA>xZI3QSJe&E$YmHUSZQPi-udp9D7EW5z+u}+8P_AlZsGhvzg_(QL?eDY(O%Q"
        "%E0^9y0O|Ghb<#jxf`!?LC6|}ux6(xWzDp<_q<Z9-GvNDl?ajh?4`@o6QGCU{v$vCl^hTHOl@_*w~y$W!on2}ZVA^N`6-YasJJ#<"
        "lx7++@p*_c{X>>)-PKI((|kZ~*lpn`E#V!z<xSUkqEwR7xd*~T*Kv(p-MXlFvy5Ad*1mI|fJU2IID7Q)!nNAV2hV|C4<21C{QCTP"
        "T-OM?QnHnuS;7nd0j-j2w&pcA`@B#4Az2k}peVTDrYMpDL}R_l{+vUK3@b|}JDLrDd~u!~y>M_|oIlQvu-?>?+4Dud2*)YRpTe~i"
        "7!{S81yz#fJ@s_VJs&eByz$1wy5~-Y{rAMCzX93w--k5X=MG#lDu`6PWc5VleXmYw9vBtQ!txaP%Y$(;CY=p&({bR}92FxSy2$<_"
        "gE|sj$`<bnwB5di8h>CtO_%dI&Y{o8pToc90-3UncIf0%HfySk)uxxOQ?ca%8)SB*PD@~&eQ1xC^ueOI8LT<et5>YnIK4v6_NXXZ"
        "stGW5eND>|Qma>IIKTMq{y{6wGYBUMn`x{S(%w&|s=r7jl8K*<H>7E);$FAeq~pVE{T$nUAr7|L<K~agBN32{?f<4T)ZjY&szT@W"
        ")z@xXC~PH-GO6}d8QYWc8-e9<6(>`3dKDksPR;SJi#GI*(IQzhQ&Nv;{9e>3;yHQj&R_<>(3rk&{{=e0enh&jn34|Tl=?VI$smRy"
        "?9m8ukLROI4~s(sQ=yJ&I2qD5k_q30Bmmd$7d#nh_MsnV)7tRfa1|==wiCaZ!6KUutvQ^fW1xs*|9S5>sVC!jBf4#$$JHRv<N%ib"
        "#c&W}(w{~4%9U*-yOG^5P)h>@6aWAK2mk;8Apq*<Bm2k#001Qi001Qb0047xV=rQ9W^ZF*Y-eR{ZDB8MZ)9a`b1!9WVr67xX>Mn8"
        "E^v93R8eo!Fc5yP)c<hgg%-8h#ADGAXjB4eL#XR`L8x-=>tb;1aCTO*U(Y#7)1_-S(U(-d^LO9(-T9Ivxjuh814+*NsdRbaEZP8D"
        "M+!ov?cF>0b~OhdMCHL_AxH#F;RX?ulm6`E;_~yC<@|DayO^iBN|PiR4u*pbxg3-TLK@*cdeDqaVSf~p+OyzDW~sFxJXjk8!@-9x"
        "al}v$XfKvj;&?cy&tRUfQ7LBk%^5AL1^Uv2*>C{-3mVl5t#k@$rLg|aVC|e~;$GZ8N{E7T3`l+k+J-hI6T~^*l_YV-;;CV!a08+!"
        "Otp&stZ>I-Rn#;_&FvO?#$wWr=%k5td``z!CSPSJRGo9BEedKA-b^}$#|3hH4Ab||+QSW6YJ^TVxb-iT!NVFiju51?thbbbuS3WX"
        "ToH7xACS^$L4^3!)CBG81Yu?7D1~Ll2g-4?tqYN4JvokI?+xa0tYn7rcj4G*5Jn5%;1PPB0THybDO?I;7&|Y0fQ>W8ZDIbmK++N5"
        "ra<|c=dj9|J83y})>P2ApLY71KYDZMXu3sxmxYwuTOho;*u{J2txYvAp*_NPVM?46IT|H9<yY4b0UMSX4;1{kcT%iNBdD6}aAB)e"
        "$bLeNn<nFvyw+?*bTj()l^@`<92U4i;`U>o#Tp^POgDtRTE~WE56=m0@PExYVUQ*srbp|8oBOBaLl68F2KF}b+$hMjQfM2aI@T?*"
        "QNP#$Plfkd9wXWW`geH<53h&nsLc9&e|h|(;+uL`K(`Bg(~gqU;^XlA%An;M1=sjJrS(37i$j98%+ZX-Pdl_5ls<7l!xhi>um_h8"
        "g|@4TFU_7(Ie%tAV);JQix%D5xRcvw$^#)$$2x*Hz9Amqs$ICdN@u(G(=Dmau;VZKggU}dl0Ci+M@Qbpq%mLu$*znJvG6}oO9KQH"
        "000080000X0JH>t$`S(r00IaA04V?f0CRF<FJftCZ)0I>XJu_|VJ~fOWMyn~FKKRRbZKF1X?kIFX>V>WaCxOwOK;mS488}*f8cyd"
        "fz;_?yJYAvU`U6eYms$DuZAMgi8fg>D5YJ7{rFLSCri2l^T{BokFSqpk|ehmZ|5L4D1Ver+W;lGC$z#!1Ys5A+pr}BD}sJPct*Y6"
        "1PCOo&ymt3Nyek`C?{8d5<y5KyhjhZC>@1f6pb2La3nXWwIDoL+r3`t+(}{9SQH|a&gOdEqAY%vNV-59zLhAC*?TGS+!@urxIz)5"
        "zH{Loh1xBM97Rv#(Z?>N3B&wCd-*_HJR6TH8C+W(v@rS?r|ay}2s<B-fIsYFn1c`02(5GqDO)+U%e>o?#7Yal;m%pyH+puY{6DQq"
        "gC}hi);0rut?(N`s45Q`H*VwT1qbB5qm0=N07M9!8OyRDXpL=-`vv=}MyNhHuD9i<z9|#);cAaCg;QI@gec9{$08an^^jvz;9mP}"
        "feKbT$c1Ej?BMp|T}!P&>~rQzdRqY=tII%rI_2XCeLV9c6sQZKFlSF*l3rn^_lzc<c{xAe=^31Vc!5uL*h{#C2$43@l&gj&)6D05"
        "PXTB=o@7Le(nvd$rpJC)6c|X$18<;t5M8KXtC8Oe#dDwG%>i;0)~JNsZF?loNv*Hw8J}8F;Ix<R=RZ&!q|_E5Xw2F-tw}XebxBUb"
        "z0%Kr2M~)l07<x@nJ6RO#mZpPYYscg#qaxBMI7NKiN)v4pdibG-4i&8?+2Z%j1#Tm^oCt>Ha)0%KiH8BjrOc3bjM?4Ck!cc^e4Nq"
        "eg(5%+5+=_<W1bxGgpRlZQC<qoKV`Xr|q)%I=h)YRup7lJ`V*~ko9T12bP2$i48x;t<T)^T5TECIx1%OZn>B(zbs}qkLPhA^24}W"
        "I42o=3J=Z6YWoY<58doH#pSbQgM+X^$h8#)3gl(T!SKP-fo<YmQ=0hC?B4UtKtREllcJv|B+nRW4SLT`D5hb68)<ffLMtc*@h}8_"
        "P>)-+y&;lmu!z#Gl+B<C&8<Fm*e3CWCsRo7>M9Oyd3&c}BkCDnAzI*2b*@pR2{YjjP)h>@6aWAK2mk;8Appjtk{!ne004^?001ih"
        "0047xV=rQ9W^ZF*Y-eR{ZDB8MZ)9a`b1!XWa%W{^Utx4~Wo~q7Z*DGddCgeuZ{su=|2}E|!$T1$fi&ArfY6{w6AU=+Hr;JwU#6*Y"
        "6L&RD?9_I;>k09<^Vm)rJ58^55+A5Et%;x4-}CB@Mx(3Kvvc5eNhQlj0r>xk=3}BY<(df&lFq55JfjJ4p@G0}v@8h}WJ4t$Rt3pu"
        "iL~kQ(ecqAf^iK>%ymiL0pU5Q1(7s|S10#SN%0rWj8Fym5?EOkmX+pdBlSkSxlG{A>&xW$h!f4$6kgqrjSR-0PBM}$sIp@5OSY`6"
        "HZEjI3ih5@b4yZHNQ;)0gmJh!{pRHA^jrM<4h_7e6mpTNlZ%_5e}8#*`SabMH<xLd!(2$v3kvid$+R&vudSh=3BsYR%7T?lTO|rb"
        "jYcEmF!~^*1}$W^2%l2UfhgeIkR2c8bPltc75VfpWA${+6q~Vv=}mNe1o%OSMNZ{(jlSgPjf^|Wm5_;}$mt;pMT`0LnekjapM$~X"
        "Z@3nm!o>bVq8P?M1&rs85xsduH!ds>jh>`6>G6kFHDsh#q&RUZXuc(Zr$Wp~QEb5aRnXMpV??ug=vvBiR>Ef!cmW7{;8S<hm}Nea"
        "Fh${>6g9n+Qpjj@Z)g!XjI~-SC~Ku*K`>PwW<+O;1Yua!CSxB>_El(0yvG!K{BZhcbcU9+Cm%HAvT@#IM~jd(0Tg?WiJG%H){d{<"
        "3lK163(5w_R*HA_;{;AWsQMO(y&`ScEEF>el~BwS&sKFo`IBD2u$O0#j>vJD>YOw=LL}i&G;-VtGOt-iC!q}YveDCJHOkn^d$8FD"
        "=a0s(?(KPEoqs%pjt7J%|I0xIKMlx-u`k)HlqOfv)K2b$WBIa&)J%vXid&IJm6MZ6p_C^m%&%Rpl)kGl*wOebffm;BS%aRWx|STG"
        "&R-XXjOZxxElxtcF@}bf6O_}#)O-$l5^%(>TV(}N3U1oge`SZlMVoFB=jj!;NYvc*k6UcQMK>*AR)}ngiHT|V+uIA*K`D_OhXGNW"
        "vZNXZ4lG!nQ|^t1v}vra;mu(>Wt{2h)C0vJk#vtO<OqRndfrj%`M?oQyy=3HT)9R$f$w{V%XN5u=AD~^t<WCt!5(a3_HtU)6_rt("
        "Zas~=Gc1ad@yK&A@tsOKH<G~UWNTvF?<K?iWE)$PNjMp78`#@U8q>DQ;foNmL)zZfRBphrwn}o=R?%#OL*JbS8V$0k2ZdK|_|&8G"
        "Q1aT|E?wutr1NKhruD5MYae@*E$T?se(h2CJ#Mnwmv;Q@Dg()*5Lhh3m$?ZWE)j8f<c&IO&K?NNhZYze6!n^Mtj`bR+Kdr3`IR8K"
        "r>q~Jm9^NQw3RrVu<+FuqCKLmyyZv8+s5YOM6UmPk)qbSLWf;F3~7U-*=wD=o2%Pve26+Bv{I8ZI{t1Ktxx3o9R9?n14wIk;I%1I"
        ">>jpla{(G_mlIQz3NHt^vg1a)z|p*I^Jzc5Eg5#p^wjU-DEpbne18*fQBk;IQ#H8U+Fa$IM7DoZ_I2*3+P5y7PqS;HkSPxG1g2h_"
        "apt-8D(INK0uJ2?)=?`yr1sY3fMI-QIr?pHhj~z&&pg;o<AdG&*2r_KHtxK_?ro)AgUP^D=?@WI<1H{6*vn&MS9ls&e*WSyHf}kR"
        "(8z%`*Yl_oJcy?ee9-BkONevY+#_+OR|UgW#|myXy&cz^+%etx@Pz)QxnQ)H8>cc8T$_6?(Pq2C7iQJaVo7<R%#hJ{8YcUOOAlg?"
        "6}P(P<@Q=&_t6l#q*c-EqGz6j)$Zi-Mnz9hZ;qvFMxUd8t%Z|231@F#0`@SjaD|}}?%XeiN7s^WTqHbqd$%J?w#O4Uq@s@ZLtZqx"
        "lHJ|nKX7Ed$HMjh#==#bgl@t`t2o5hn`6<k@7X$?7Xq^{#*ylBQL2^H-TPFnY9c9qDJkLhnzf{~GAVf{YwRgsgPnkJss(c8dQ<tM"
        "_eqMnQmqx;06#g|#UeC&nMztwqH&4WRAMczY2>8s7KCwS{R#9KHS=o=a_#vADd?U_*!_~7z0+SBG9kUf#s*f_MYL+sMDVu2cCj^}"
        "j$gzk2Hngbkm<gpra{ttI%h=@{nNOTfD=0!Gd>^1!9sGe7A%Kra@}qXaB=+$&ckeJALe$FX(nmmnlA!V5xhm_u(H<*DV|lj*Tv@r"
        "E-q1HMoRc*H%F3&CuvyTQ@m_X3$~<@JMitrvYHVa2If26`~-@^<2?YJ?nyyAGjkj5ASi*N1lGeWz1nY2_E)B@ThqJQQ!Q%Mzejvb"
        "|84VNpeuM^n;$oVbvbJ7L%M$fP)h>@6aWAK2mk;8ApjneqI9tW004sq001BW0047xV=rQ9W^ZF*Y-eR{ZDB8MZ)9a`b1!XgWMynF"
        "aCx0o+iuf95Pgr7e;D~Gk&LMj7byrHDq4!v6hsj`AY`q*V`s#A*WF#WHR9u$^(A%Ev;r)}@ntS&&Y2lUQM5jKv4Ywf+2o##hC<}t"
        "*cO~JK7o_lT2W3OwD5F|;7PYYRjsf>?M3H74%SDnp7s_=-6l~K&1dr&n;GkYqvf03yF_ac4zvztg*6pqRG3_-601rixzPocFnBNo"
        "v=q`^)|fwF?=8c2qs!VzR395(H$H|-yn4pJW8njg``#`WMPZa2oljBt#^QzXpHay6#9Ctm&iO1?!Z|qZo!<xNF402kgfAL}%hh}a"
        "d`K)HO-Yka(}hD-#NZhFYOj{jJRGj!H;ZMmbz4r}afBoUfHz&7fMraBI%f=Npa5_Ekc>1}o|1H{f$uM<ov|?p?>RS%mR7?3+<m)*"
        "CvSHiOwh5Sr%hmG?1MAT3*V39bN-Lt@#)8F&<4uDqZhVB55|IQwWuf`Kw8K}zQMjd#>iHX!-JeJLTg4bMBj+>`pq_XMv|X&-73@="
        "JBgN4S|oO~C{vPmi(M+?yW59|y0hp5*#mq{j0Oy;<B7n%SnwMzjK%}j_e1X5{+%F=?pT#jA=55Lz@0dJ9e-tINawa6Sk-F)87iYo"
        "$KiQ)U;dvYgB(NNZEMBw>HGwQmT<m?4g2m%Qy6<AtQ=Z33uC$38Pcxq$p{}Rd-Z5DgaAT{+B2@RaI?n^KiOjY$I4F|rJ$SrZ8RlX"
        "KHrcfU4~`oJh6?%tA>${)z4?4_N*Z#)7m2ME7B;_s#_ojmqS2#Z`TZ5+{r(z>x3Hz!fu-cXQ>$nch!cZL=Dp1Xk~;1yd}rWa)Wj<"
        "jE~el;eUPfJh1R#`~q}3Uo-5Y9Tf@ti-e+I{nsHw7M!~zpy#FUu*wvFB8C>9&;9^VO9KQH000080000X03!&8-~bN*0HiYj03iSX"
        "0CRF<FJftCZ)0I>XJu_|VJ~fOWMyn~FK%IUX?A5UaCxO0+iu)8_B}xUft`Jk0$H`)w9UFu(QP+fESj`I6ZFL}1dYU*QLS4<$@aA9"
        "e*2ztNJ=C{joo$tXGS7<c<whco6Y9OY$)$I`wy?{jJ;$-*&JEf41C|iK-*lgK5L4$X7{|@9|jRIp6z+hvSHv2PKa33vVqrK+h-Na"
        "v;4s0=g*!$`*7fF?8MOXtcLN2fr%l@Z&`ca2=?yH>xlJSl*O1;0Q2pecMP^+Kez8*$Ly!3I<kW2ZDDTo{8=erO;=?(FJksStNGck"
        "?eDX`VB(PVykI~40;uQJSa89%?Qmd%S33fn_pN|;Knvg}0Kv^>gIJb`_rQJ?2!GcDo$^gpZsVE{S&<Eydd2Q|FG|lmpral89QITb"
        "cV)$enl>D}vf1mAJQIVl?a=o5!G4OHhGl{^dYb`I7C>eVn`<Z}tCt^ur{A^BuG~k)o43HU%Ck1EaX`d-nkk<Eu~a^5$ieex3_dfs"
        "e@$X`NB%A4^Cmm;UJYhlR~^&79kTwIcCAQxy;VaXH=YgY?HxRa$N@rdTkhICs|bfUCt5TnuOB+zmo;quzT-J)2qOHx{Ee?H1zPiA"
        ";t<XLX&kz7h}e7n@e=;N&4&CCu@CJnZvam5Gw-spfBlfvUFC&}vIo#rm4Re5WVz})_13pcF&db@=S4cHuLiIk%c@9!ZNbLUyHb?f"
        "vMPt84-k-4HXx(8-TumRw9<*b4rxhAsZ#*ox7`t$`uthJcdQ|cP9;AB0kmGt?2Fioh>ZLSETpYi!anpPyyXpvb9(zd*d-6Ir1Zd8"
        "z8|ZK6(|t20l~oD7^oIQBLSlm9byPnwqUzou^kAb?T;+0?z5v{qX5)#)0oT-L6$phPr=62(@n@eN!aG|_~KQ3zES&=24$t-?B@}f"
        "S@YL@-}Zs_U52}2;3Mv`ss!1AKgjmLK`~f{kR%dzya|=o3c&}Un5;I~SR$Y?z$r#~0n-n`ZX+!w{T&$_SM5FTgYb_SJ&1+~2#i2*"
        "LT&}7WQ7O_61a-2NYNPv#BUI8vKfcn*{e+$UT<(fiEVQ;v5AB=^WnZMz_BIlvr8OABv=9rdwu!shJoI3_VU#{Sz;289r7cX-_sb7"
        "xlIv3a&h@8S{qEk4$}&)JOvhvgJ<%gEDGL8P=uZ{j`9KC7bPGeXT4-Em~W;5X|eB1cP{c0G%|6~*;nm0tI}!w;tOMfB_hoj(9e`2"
        "S3+tZp~9xyvM770)c|d><7AUn)dqR5Oh$73$F|`LwsK_2!cWI&<A72=1ag0CL5zDeX^HL#{hrik5G@7VjrOC%*f(m0H2J&&-D1)d"
        "-jdUqoOa-G<c3YZgUH2uFd*qMz^n8SR6$noq6P)VzzHE0A~x8~SGZ{o+=9uRWjmZk;gw>x3a1DdU66{)hQ{GkLkwEbKqVK(dx#n("
        "Vu_)H7ib{n=p=-+&cv-TAmcxT7Y{@Tx#=xGrjq3e3d$IMVf<DCO)ccy1$q*B8Fx@6E&jn5FK&S;6G5S2)Syicp6JM$b|AgFvh`bf"
        "!e4Nj?FT7$lW}Ke6)RdUL5Z=Gd4`>=X|JX5@G=3hK};BUe2W3*fJM9>cno)Ym=yD^OF_>+j$n#{@rSM|^KzI>Php3OO1nZNWvCh="
        "-q?T*8n-eJsCcsn|HD!+F^kq1l&RyEZgt>eS>P-@yNF1uW9?{+J@OWkT2Z3;dNg{T_i^M9B@+qfW7<H73e@keZ+{Ie#KQQl<o5yL"
        "6NXD5zVs`g!DvDVTw5dH9R~-CzVR}QDLBG(9sog?wlWYdr69wSCPp6F5h`eHBt0sU;+Wz!!H|v|l3{*hm=ZWn2NZ@$P*GZQM?5eM"
        "VX>`d-L1j8+dsj&8H|^CMA%<p9eyT;NLpqD?;b_)G+>$bO=`KcAgQR!_g>SM>VAE3<5@uHs^~O_WBJ0AYgs(p#LZaG{)?ioSOcjo"
        "CR#Ppna<Nv_L8KSY4FLFxg2^-LVpps#8_5y5HJI!Mw%K|xM9SkI{DSdfJ3h$6Zt2=(pD@05KniI#;4MQfr&cC_o(Tjy3NLJYyv3}"
        "B*F!Z`^%fa3L23eDwG=)=)a*Ny)T=hy*GeMeohsn#DJFAD!1?3u@lBZ&BA%9;^0#ji@fbAiI)PXv!Na<?M`hmVk9^N=0j2|DS`R{"
        "DDl9OKVl3~z-XvAs^EW$cM;n`r7#4Wv$EN3!gWmU54`7rS`}x_k@mmkyhG7rH3wQC*F<W`ZyZ!6t;$;-$N>{PsQNY|F_k8cqTLO3"
        "_CUZ3lzt*%5tg5!^J^tHVj4XG(0X2>{rtrGripiBqZ0ABfMT~D2gmJMYa-Sql9?oqIcrpFF<Ci}&w)+4#7=0Uep*xu_yL}b71oJ?"
        "jo&xQ7+h1NrbM|P+ff7|_K46`EOQ8J2g>aUl~#HnxhWtjqo14Ze}IIKD*FI<1V6^gEZKEdIQ8Bjho=;Ai`}(d29QDNoVp(CcPP!E"
        "#GI0?wi`ev-4s8CvaB4^lz0pZ*YW&n(Y5jQ8qB;GQc^6R;|Z9l_GC%`26Ym?oDtn0Ox)&I;PgLWsh%7Q>VP0h3H}qh+v&V=5mukD"
        "hg$3IF|}5tX*ymWy|b~{9-InGuR`iZfXNLVtc^O_Pa&z7YS|>t9y(%ZW~^5tFeA0l&^$PW#%WYM=?v)he+#r=h$EpYQS`QnVgs=X"
        "CcY^fp7p^rBT{V!wk<Out9wTR&Lj4;AOP_PFj5SB@BsUbP*q{0z~VKHi|}fRv0Rx@yEj;AnaWAwa`9aEBr8?DN8*u0`V=^7M9Ii5"
        "dk};`jKHY#5F0}nL53cFNu^z(a9)cE^xUVnq+HdM3rat(#G_>5)7-i7v?4QFl!EI`@7ul*L1;Op`903FT#bQDEDtkW<Dm*I9k4+x"
        ")RZS#1=XLEyNVG9t=l@$A)MTF%d{BJ-cLf$pq)w><&Jj^p0DI%FL<)*bmanbs`@_7ZarkgD-#b_%~X7tURbIXVZ~SOo5bojOXw=z"
        "YIGNHmk{OG1XF(fDPrWVnw)QNk20_MaA*sI)4HY?0cJBc)&z#$1IEoB5YNY2j5Q=uAv<Fi@LXkem)2zyoLkB(RRLoPb%-V3sP_p4"
        "79=%x_seM?xw@&FGHagpiS3z<@>yq;7i=?ur$(F|@QA9_k<D!(O*l`rB|7SJcP}H|Iiccl!|$OO@4)#l-H^M}`3<C!e#f#JiwxZ5"
        "94DAL+fG|1vuK<ih9#$DwP}f;sXK%KrFzcF<p|ta>Nc&7BmqK`7_72&9{dN$opl*?qICa^V}qXk@aZp+IzPoRN>yQylEM2!TUD~R"
        "24$X7(b%86F11bdIzlJ;x5$ezMGV60IF4nydn2oY**>H|)2iK<gII4Z`^2K%up<GD>FG`{N%2J6n3zpEo1&Fi;xMgLc7H*;e*fdI"
        "BwI#r)^5sr5)fzN<ALO)m04W<8<s{4Sw|N645*s2BDAP_4FWF)o_yK@=jmG$T-iPQByl*UKwn+nc$m6$j<1!_Xsdb@1N;J`aoW?C"
        "d$bgR;@A`Wbcv1WiT0j6d@#xNOtw!dR_wzO82Eml4o+#1!{k48gyD}fdDb8VNJ}`DvM2azyKY6#-*w$5*mBZwo9mwLw(;Yf5s?5+"
        "d2<km<9-}YPkfO-uCK7^aAh>%r90J(Qwg$m*v5%Q)~(c<yAsbw2~3JjBbeD)O%%WpSFdZsgo5la4Ow#t)Voh{DK^&NYY}ba98ZX7"
        "@E~2qiA5S^3Dp|{mY&~};u)-1h9pbdPH$=N7A>L)X319lKU1quStW@Sudl4zE|743ZQg}>JY`(Q90EJVbWdvc$uhMgHZ4s2TW!_T"
        "d0qy!HO8FTuwev#^M%Wl`WP`4Bczz@wuR?3$(PSf`Sl86kwdYQgxMP?bWW{Uv$RvHBjIVSS)ir|oXpZ<(n=4LRM*f5WoMD^M$t4T"
        "cdfE5us96oi$=27uUcW)>q_bFQ{d|bk}5h~OC>FZmD|!pD8^>}*yyc9-C`%(+i@)TU77PMtO)V$MBa@X+8RTx+D|w1n-Tn$;JgSf"
        "zSK7{#v+^ikOURTuQ0|gH(j;CN%Jixd@|Hd9oN81?D_H`?Ra*Z*1T@}BU!xA0OcCu*mQ2<=Oy*ptyM~gQJJoT%ZOdz-)HdmGK?6#"
        "_&mIEn~SnZc8$%QAXE{6L<>;tYi&VjDCD)RoBsQA!A2aL;lQC<KD*dN6M~eb5klvmN9;@7`r;Kl{S)7O5xT+gap0FG@Cy_8D<b`T"
        "30RT3<YG%E0MtuX6EBQQVf~<HKp`*0@*LV2jJH{SOTJjGpRUrNh^kTy0qqgb?*pB>?)$7LL4Qe@U5t;^h3v5PXu!ccB68HgMA5dx"
        "_BkGPWl&gAAtm*&6mlfkhcOO?A%Wf_=xJYtN7FW#X;a6gmO%D#w#Cy6d=av*A>CeI4(K6L&b||m(32<}Fi*s(3TgGEwtw1<Kk@*0"
        "=u?b2)T3o1@0=(PizsUBrbwc^jFWQn$$cbbDtPh9P#qLnvN5ev-iF>-wI!-lBELN4BQ-k{DgO2TIP%Y$Gk5<zb*u?<z54PXBYW0K"
        "#hbv!NHw!p&^Y(tumI)s#mx$t<RVYY6S4+NFJvxZE|}FmLEY={u$H6hcnrf_H~EPatuiq!6jTkA*qA%1NGcUEkzZacQfz{xN<~w>"
        "GEpylOVaHT^GO@~7hu6$-tHp-Jx-9`$^f@F!0mnD9P>1>!&ro|DaC_Vyv$c9z(&}rvj-iordir=>VW4}&KbJ8S9lt@p7bQx*snEi"
        "P^sHUiIX?kn!9>&2;23x_WQc;u-w+`lg9(0&q8D|rID4;oFj;*>|*j5+!&rx!xZ%&GE(e|19Z2V+yZTfI_qy!{=oBbaHg!KaqAh}"
        "begLg_sZ{4isio|Fds#K2UYRVkNLplom!~z@n#|y0FUmueub1-mr=pCI=mn*FvkTPcRT!F4AcE_n0{Bb<6JfFA=gr;+9@HUL-Eib"
        "7x|f3Om;6ZP}kF!2l~6%zU0+BXr96}nAB6y(d5Cx{{v7<0|XQR000O8001EXt6AZ;e+K{nO&0(FB>(^bb8=%ZVrgb?V_|G(Wo>O?"
        "FKusRWo&aVaB^jIa$#w1Wn?aJdA(R$kK8mAey`O3V9ZNWrfIu~QWb)<l&(~%lnOy5R8^6aI5TFGIM~i~2gHBp9A6S=COcas#ICeE"
        "6Z_oHcP^onzPuS)Q3*qOX2hP8YR9YVL5kjxR<c_3Tf+L9u$m2qtB1004@C40SG=z{QCu6TfKv7urS$&2`}agQNM*>5>0K+z>HF_m"
        "_V@2KN_J#m29Jn8c?(}7Lf}{J{7uaoZg{Vy(wTi8Xi&AH5iLhTd$kmetXR9{UB|Lc*1Xl(Y)fGj>osT{7=-VI;Wwt3FP`D%9po95"
        "qiE~GJuB>uHBt~GdhlV^@lL8kF(_G1ESzOUc`#g`B_1S%=n%*%**9W4DqgtfK|21{ym7v1k<CKA<o-PZKYHNR#(k`L1qUj!-XB0{"
        "j}G!TvgiKv!9>0n70+ok){KT{<}mQwhAJx?(8N4>>W_@*5BU{qb>Oq(tX5KX#a5L0v+S{dtU0gqCtt(2^tpQiiFGx$Epc&xocm)G"
        "ijx<={&4wdP`qiyc4xjJy(G2#*rWNHNJU<}{gptJYjA7B+O}lXwad|U+evOSF*!X+Z}UnZIXy{<Xx0dX1~m;qZxXPD+fR-?5a7|N"
        "_)KV<>Pd=L`7nl1W?qN|>53vK(MkhSWPf^NX7B9`ZpR<UKv41uj^9XgCC9#gsgzU+{a%_~h(t|nwh@F9(Q;H(Tt+nMMh${*gf<yX"
        "7sj4gaNcDnioIj)$b&{q^HzYRRveNjUU5V$>jU3Ntyq;f6Un#a@mV=<_|_GIKy_>sCVIpHo;o}@yd>^lnjuoiXE|~B3Gs1P@e#&_"
        "?U`s<iNT3DuIptXvQoW$khH~%Maf5wfYAg%0dX_Do-FYC)<L%=kk@rAn<q~TJq}o#c>Sr0qIMXMQ9&ay&{HNgCsyS9E0Bp=Px_U>"
        "Wl#=zrv*5%wapb8{Ii)#RL2dJC|{lIU(fKiHjqq`G7rejTDc@s`6;k5=D(<s&&d#HcjOhOHCJm$j8lESeyfU0KhFw##+&8&skoyH"
        "`dSRdN2zL^Tg8pwMIZ{M+v6j;6&FM|yf<9RxmE|Ui#uQB>VYW+q2fwU#R$SWn|2OnjpBSm3X;d^30=3?a#O&-A_MRinT}L|_=S33"
        "5cKY%COjY)4<3B^_|Dv39^(&v)$6foV8VhlcjDgteEB135KFF(1UL@usHdW0f$9NjhpQ#)kiOQ`j`brndu*Hu{_%$v6aNk<?%x_v"
        "IyMYgO>*)V_1Qq5-=3RR%sJ?!fB(lY@e01Do`8P9RQNRPIBow&+{6*e3Z$?%68~IrGpgPlZ^&OW9pyIzSE7Sl0`FUKJs@Utbfbdm"
        "S-3{5!#yL5T2v-U<ICP@aL0d&4}D%1wp7<}=Lp)=Nhm}O>Iicv_MloL?A3VATfGYb<ZzF^wvz+S9~u<hv4KCj<_EH8t*CA0!%W9r"
        "ykT|*w2Xr`in9)!v=<sL1fCg<1!7)fhGKe0kKew2S%4U|uH~qCQ85Vh(c#}m)3t6);Wr`}o}iLV8&Tqe7Oft&P$_x%9YOcc6YV=^"
        "p`VoW*ll1+?L>QcgyC|-5IFSUYb7Z;$idsJ-y#qly61{HkG4=grpfe3jeuo!4ZSzFJ6UGu<s`9Gj|lzw<>e2*Wq3l<3>;ls?6}Di"
        "<mPlV&BJFj)Mr=6I@oD?lF;CjELz;4??RsSqJdS>^NmoByC7)dr6Z^LtQK3Kfm6RrW)QfGubw7I<5Y7EBu%p&zj3#dWqo(1X3zr?"
        "<elPB<1XGgrd&E(;*lzZigZGE%s}a-lS`d}w>b64{!j3$)!h@J_7ofqBB`~2#%rR5QYq!N&C2}+Jv+U=npSr|flYH)Was{3GOUsv"
        "xLcH>1%!chL8<q2aY8>#YA>EU-J_~y^>=k|Dqm+64q=#i%oV}Lz;g=<`&~I-(<3a@esPZG_OM~=yZ4M2v!5U@u}$;v3MKIO>?WN)"
        "CqLn96N?VQa0<{nQo_ChT^>PBG}J#o1p@;kpf>#J_7>cl>^m6ehHTQ9^@oH)o+(ybQbeCcVSpn}Lc(_t>*Kmu3|D6@d<~fqz2ga>"
        "CQxf*<&PX6#+ZD5Ig>!Co-nXl8`613uZD4AC7>uQmL@mB$7<iP3amBwXzE$aCz*SrlI;OgWuejac}|{OJUdz-PWrb4$lBYOd8pm^"
        "WC$-2e0#=0%H}>X;{cu-GTZ#pD1rk<a2}3Nqi7R13ji**iq+9n<<_wv&*5%SQEay2a#<j2hz0I<8cuv9EeU>~czhC2Ck?F``m|R1"
        "<aKs?S?}@ghu{O0i=QHAxbLE$9GD5iyE{|)R4K=3O)0D>FmWePOWVX{pU@5BktZ%+(+tm37bM@8K_zcPwnxEts`LgE2AKw6P4x8k"
        "o#&P;^TWm>!{!Vjxt%$ZTdTkc+InfTXtBi4F*vo@$Ka@!&N6&hD>)nr??V_wlamX6HM?Uh_G3D@`Djp%CuH9zpGTd_Kl%_)dfQf("
        "xHB0(Ebhk77X-(><GJ_xJR$xAP)h>@6aWAK2mk;8ApjLeczT-!001Nq001Na0047xV=rQ9W^ZF*Y-eR{ZDB8MZ)9a`b1!mbW@U0^"
        "ZewLGaCwziTW{Mo6n+no|3K(T+7vZTvNki|0qfun1)6q9g1r=jKufgEMItqlO5zpUZ{InjC{c=<bOsDaJQtqJ$HPxi6kVO3FNxve"
        "Kok|FtWO>o-|Y>dTC;jn9mrNHVweV%9cef#m`;h1pg?|Qbxp~J7IzydSQbUm@zL=SuN$chF;W%#@mnSYxFtfM+E&S$Z1`3dwA!Jg"
        "tdL^MccioU?fTQZ_aD|bDOo=@O!1ltbJMWmE0#vl!f0=NB}-O)YE9Fcl-#ht&fx!+nqr@lTY1L>|C^Q9tf5@3A8FlGY|d**4Z5x>"
        "3ekJd&gz$Lb<V8Dy>$k~V`T|N>w}_AQyuaLsmi<;=8dd)ad^g0&D4&SIW-0fz@-=h*)qjM!LF>r&t5*~3W~!8?3T&RpR6$ap7lXB"
        "Se3j`QtKQXDA^paJI6;w1qdLw0E}TLj3^c&gNL?a$@2J!z^`OmlIO62$@5sVYMT-bjap6`my&ykXp?Iw{*rKE$cik^F1;80=@ty)"
        "B<sB;gN$#j*9-!IFrXird<4`5*7$=&H6y>%s%2}Xq>3ZVU;s1hfN5)D_6(5SD$@BoDVg(oPZg)a=qw60$E-5k9%9xOgkJK%OS9Lo"
        "{3TkSy^WUS{46EW`T`#>F8>)6yM6MK*P!jpK8G1)SeQQsgD#*MgHAFvutEB55M~$_bZhOw77q<b!+26afkPuNA6KW-ynP@qx(p(I"
        "!DUtFdqzv=>;o5!s(54_y4`T9SMO-0+0@PS7=gcM4p}^g%F}YCGYX^R=M=hBCL+UG-dEVUSlQmq2%h*(o}#9r2f(H+*DcIx4MQ0Q"
        "!`ewdpm37isJ$1^8(!yI#!4`JpsHM5knhGiB^-U(n$>A`3MIEwVat^^t6SBw+1SDYel4_klt4-r2t2x}ST5j7fw=}#OG`E1-Y27P"
        "_0_5OO!i|*h_bOnw62qSbvYxsyYGblUYa?InOB<F;~H)_ay(5|+S@7@P(4@7>#LiO@DiSj1P8OrVoSPCb`#fjY&p8?5_0lWcoQzC"
        "u)^~fHu1#+65Cz+H7+n>Ug~6eFjSG>u|w`>Kuz`C>ff<(-&HG}FN{xJ8ww1>M>{$;T}Xay2x;(QcM7td!YZCeh}$%s1J?Ake`98Y"
        "s^8;VicrZNH+}NmxQVBji6^!CzE3*|d?{vHB|=cd&V4#Z1bUeZx}qB(BuV1<0YXr<B3J%I=DG%I70+M&CA;SZT>IMKsECTDjVwG%"
        "UU*yref%UC(FXx{v*CpY_$wrs_>jUSX!gK8NdtKxrGcJqd4+@&xf%hMrVFW<c2pFdI^J1R(b6i{F#qDko30*~UqLW46bExiG9zOb"
        "c4xpGRm4_&9!fgTC*!Ek<75$~BnDQn!2db?FOqKI^oNwZ!I#A)y#0tPuaodFz3lwVI|o;b*P-J}`*J#V+R+8uP&-aTHXA3g&+%#Y"
        "4r^dz>*kJKCqi}wRMKf=6(9)N8Ga)mLkrEVF!K%d^sb?b*32-4rLDL&vE3sZ)?*LiouVZNRM{r<@(h>lhf2Q$1T*vt>~W$UMA#Jt"
        "bZ})OWfeQzaJ@w$%HkmUBte|n51^_FM#r3)k2c9r7c4PEzlag`={P?j7j{hcx`pvjdoJ6<loYatfT+C{8wGT{fU_X*0}3}L80f3s"
        "c~51#<NsyYC1U+S1;T2tGyzo#AQ)jUKsI1*IPy`b=g<P0?jaPpIg;x=ZBW6sxGyP+GAjy^A8d!ofK`Yf$gz$`>SM-d_fo6`GY6&7"
        "HI{XsxBvks9o;CMbevj8V%+eZI`5YMw_7{IEEyV*id|^ycmOMq4>)egbLbuwVwjD&+fhRSX=%f?EGtyrKwb7VRd+diWJPP9Xu$`}"
        "e^5&U1QY-O00;m803iT_7_N^!1^@uN5C8xo0001Ua$_%IX=ZO@VQgn*ZEayMZEs{{Y;!MiZ*_EOZf7oVd6ie&Zrer>edkvU)Q3tf"
        "E#DF&P~swSo3uqP4eZn}fk7|Hk+cxWWp|ghm7@RNnI)Gu$w_O#w#1#C>zOmd$z<}gDl9%&D22AtN>#{W1)@ml7i5-I#%f{Z9YTtU"
        "N>PKsY<ckZ8qipo3#&9N>Gwh|RU)!qG0(+32tKXRAXHXnq!SIK3JPT*Q-TzOlY&?zn^+YHbY13UhB+3NPBoUwNK(9oL?mmR2Ul!T"
        "SWD;RptnqF@gd2o6akCIH_MMhNVGDhyAis8auUhl9_4Cn%{-V)CPA>&Du+~9kz~Rc(nyXdZM#Rcgx!{$mgevmnb;8iC|R}0LikjX"
        "^Fh#@lH<ub2!fZL4)+%>Rb?@~r06b$P1cJ5=!d+ZCz`TVSThLzAj3QIH)dHN2`Uh#F4w|r^Bk>~i4Xk17Y83Oso1Z%3y&={B9Sc0"
        "%0@CZi`vEcCoGK81DQ5sQ7+78k6>fxtIE90H<h(kjg2V7NXmRtKlg=i;2b^>P<Ae*SlabBZv{U8l-dUEt`xehz(<dL1Ewm=Oj4q)"
        "+F$|JsWL9@rB+lMI{FVuf5}Lc2~}yW(k3fQf^(Ks89_AiIpSuDHg}N`cLXpqb)x)3#`4le85RqW)E4J(e0KWmN!SXku;`yYJwEBq"
        "iYkvXu|;kC@#*N4Os-L+?fJ9rn4d?f%>B{H)3fd{-co->C3#M86rD^hF@uAjTz7mX)(Bxq?QU-t?d1uxP0lzTl1bOuL@TPuw$!i2"
        "R5`a?RS>c09F|p<^#tqa=0U6bb<G5(rGm9gQ!3yii~w&BuS3rTWG*3!6Al4gtd0b;jA)HXPTisA)XZ!4`rg2L8*51-txmo`5ty~m"
        "jzlR=BE43h;~yjiM2-vxQDxlnB-Q8H&=IBQ$V3%OxvI4DtrzYqy)s>5$AIyiZt$;)@_|54S9T3koUi5tqzzN<>B*)~wGQuQHXkT$"
        "LH4dzpqBgy$sK9~&NHNPxr9TS9?Vj;euCnDZGBjNsPdSqL<%YH!f=i0tA~Xz@A!5H+wT4bw+DA3tI#2Lbg2XLZ#5q1Gdaf9Dj6hP"
        "(7}ZrfM4(fC(JXi8B)`edg#45K0AE#@%U`kE6pX^N*AN*Ts1?}$S*u(3V1IsU`nnVsAy1HLWHT7OZfn|w^1J7-p=Zo<qqn*1AR0K"
        "s>C*7B;=A<Ap=X2a^nzWl@%FWU%q+|L@o|?_X?0OvK?sh7%`-i9fvQdRpTdn;J&uz=KW@FKg(qge((a07+(!``0FqiT1X1{noIUt"
        "Yo(`?t07Je<kSujbNdtpu3TTtC$qjiB8O>*r*MFduq}HyuICJ1oPX1w!KsV&9a|y5R|O4zDr3(~hp4(@9wMB<HV)TiE{K8dS+Iv#"
        "UF*0{E;eRY_+bzqef0s-$5Dix%J1OM{p1-7f|M+9(NxX(-Su_dyA09>d0o-{*kY27oMCcr+N=GKzU;Hc=ej-vH%9^PSDweDd5^h6"
        "m>wUUJRybrZ`P0Yqo2=)Cb(VfM#O43tkEX>oLScKCAAJ_2JhFXu?sq`SKAd6G7s)HaF6i+9qEon7wsc<9*x+w^`1o0lY@OG+KX*B"
        "M`gm(d)jI`9BZ06vq$qI?zyoQ&K1yE(9`TF<nIB>Cxd%_<H_U6VyrP7-W$g-*qGe?JyDKy^cE}^B3<$yACc^3r=?c9HAq@_^`Ecq"
        "dm4G!i)AV`CiRXQ@yvA_amO>n!R;;=_WI#YCg}I(`de?$>hHRP9qYZ&8(v(xU3O>80}V`>(gGxDsMEeuB7rRQe)OlAA1Q)He45f)"
        "bLg&0y9JKHZ!4vc`kraDV;Mf4!|$}>6e7o7R3_0fCSdbyqkyqUyPcY+;d7Am0;aH@W2rPE&X9KJ%FsY#t>i_kjs}fke`w#jmqn`X"
        "nYfIk&1C$Vxqoyuh^>=JQfAsuL<lLrn4H(gu~PK5R~O>E{x|zt`P*0Os3Dk$h&Hl9uh4eIXL8<slj>mL*o*`odhl=naIc+%SDtX7"
        "Vz_x<u^Qv7b$cNxRBh@#3-UDZ6xnbn*R%F}-cMfpYMC+}`anJ}LRhq3-|X=Fu5=gT&A2c<6JcLN@GnqH0|XQR000O8001EX#sX1W"
        "-V^`;NL2s;Bme*ab8=%ZVrgb?V_|G(Wo>O?FLGsYZ*ysMZ*qBGb98lNc`k5y)jVr+<2I7tQ<eX~<0mqa<i})^iTAX1o5Zs>b(fi`"
        "@nq|&92W~BL5X9ER0z_xGUfk%-3@>QL6DN2>}{P=6N@Ce(P;D=P5Hk6E=}{S;9kXdrO4{C+IymIlD)TsJFH@LRu(gF%?mESd#p&j"
        "0>0jH@5lGQcp@&VtXO+L{`Z@O@B3eV@#PoUc2`!l$Er0T5WKyyVPcbKtM>YnD2w)@6zx~Bhwo`sZoRn7a~|Wg!cz}_ESsX{l^U>P"
        "HKNoH-@_$RK+jfLTE;A2^X--`;<8AywH|l*KkqL;y!rX^?Z@ct-+sQl!Ufh5f7tOV+w!6w!BAH$u3Jhz0DrRT@_}u4IiKOBV&Lnn"
        "e|=T3o!FF4tO=_nYW`4b#4#(%B7-53P>>O_ZWP?uI)x$l@(T|>SMPrL@Z)9l;qo7EuHL+Rdo`1{|DM&Fq$_~p>bBf)9(Cy8E9`Ef"
        "o>!bDktmxghB4*6Xm7W}YbI_ZBod#)<8YMZy1eB@_PdlQMRZ;Q{yW%#G~-F6M&M54e+f^$mK!t$N>70Me?i+Gdh@?Jczz*?#7Vs+"
        "N!$YG5d^uQB_ZnrUc_Yra=i4LI-S4r!%(lTZmPn8Sg}|mzxI(2{>@BXt(f3ZTyD2njW_ECWF+W|NbuEHDf|h&w5&Y*=4FL~#w+39"
        "C>p|!1@~1}b9?D>&QL9t7KR&H)QjzHl2w7c5X+BM19Hh9GEqn6EnS8NjKodcAU-9{b|(Tv4M2rxKs1<$vuydA<pL08MFMD-=b`tt"
        "=l@#xILcz8i}1Whxs)*RXA<9SjPEZ#m04jf2L;oy8odVE3mAi2p-CIAYp^X!E0S!@MQsYDTGoA5ZzP@Nju(Mnt^Ckq0xV*Y<ou$e"
        "Sc#jmlnaytja81FdA``N^OxWH;bOxd)H8#WivQMRu(*QXM`cs*8W%lTDyboMiXbGT&WgR%3gBdgtPewzGL{MMq1;{)&}CJXRp7s)"
        "DLfbsb{Z`Z%TcNJ9@;dPf!$_a)f9j6;!Rls0_Fi>o+NBx3P{T9c++V;Y>wbSn%MDp1{N{ia4j-lfjZ$ZwR4I_@ICK#^&IwX%WH;G"
        "d(tA4$UTs~{S4*@8{=)ugy-Hp&(<5DzPvBcDo6*0*vM>cn)MoXJ!P=9o2GR#n-s4i4TK9okS!~+6gH!8W;zB2NPQ4Rw5L(1q})64"
        "e7V$)+(@HZAR^f42{YA-B2kUP%To@BW3ndK$i77af}HT`24p;tcL)vM0YNZtcpNzDUEi1+k6tmZc>?RwbX~WTEQpU$mfRTTu$6_S"
        "+&Elg#qNFhI|M}Ul>>(Z{{$?EA4PgiJyb^<1I>nlZDE8|59EYiMQdA8%C{i;#!B%cH5n+U-ReKXkG1*1a6vOQcg62A0q$+t4biu>"
        "0Mq}&{v_6=a$J%uDF~~5%|&31gN@WiL)r&2+&vrN{o7J8<YKrksowu!dBc@ve?;Oul1)PGZ8NcDphpYenmg!r{Dc-CDswC;B|<ty"
        "Z^MkH2p~jkO+U&Zth6791$Lx%eX6}8tQf%;mf;Y_;wEhh0e8<xQWa_(hjFN>8JX~kvi9DV1($<b(3uwj-6PRUU?Q2ER7I#g>p{aQ"
        "9FOGVBO(@QTLBLND@$8(QRa6%Xrr2elaPPZbAe$Yps*uFNzgj0CFQW5JTLD7O?_Whw@Z53yQTN40Ni5cZL$Pn2GosD93$x^*qWL8"
        "r<^%(1tTeVq#}Jyv)~|(pb5MV40DG;$O9l`<g$6uY%x#afgP+$%JEyztAP*^2Jf$mUw|meGl<$eK%RrImZXR60D5RC&<bX|?QYf8"
        "-Wq^`G0;91Dyw}aYH2M)<;L-goE7VatvPD4zuVUv@D4}mGUOAH98VM53XI`^;`=pCgx_x3CF^IAYwhj{TM&WS5lE?obxch2L7Y%C"
        "u#jLraT{@~kD4Zduo@C2Ff|4S$0U-!N|J>GH97^fT(C(FqmLiUx&|$uFmn74AO25wjJQ8kY7cQ#KU8y}2PD$l4<9=q^iohChDlL>"
        "wP2I=>MTs?>>a!?k)zVLsA$a=4vMFDRdxrP005&oq=T!7Z8ko=<56dFhtF1d8Q+5CrJ0v#C8WiKRLpz=NtFYb<r(aoSZWXwU3C-E"
        "w;{lmJwz&Nm$qI=<EQVDv|DhtXP2?baxC2s^pyqI4#HR+z?Hp+7^~fKjI;uUzZEPCjgd)??*Yx>SpH|X5X&6+q6H`YaDdZTGeb5&"
        "jN4<7GJ?DCu0Q)oAACy}M5yNz|BIS#sqc?6zb3n)Cm{^3eSMtXGm!RcztcFt;@`j)beumyLtt~>GDqZ47HkKZV1oIm!x$@lGGY#g"
        "o~ltUf?RYuVp;OWni7T1mNt$fk|eP0v^CD<KfYdiXJ#G|^Ih!)o&BD9@2=FBj}YyFAk!6P#vH#{i&jyCCin?r2>}VYd5{YkFpLZ|"
        "y~qVR;su1GLGSj=xDBqxWliLqoZ;SsWY<+ds|l|!UY?#hrOqjIRhgqnwTpPyMzC;meXjP^tgo~dA@FPPnFUE<?xSQ$oFK9Ez?b=+"
        "&+owaD7lnl!<mxPF4&gRVr+?XN{LI!l229%0$`0+M}Tb@N(uGS+&EvZ3>Y`LaD$6JE(*5`{+PE}i~`4TCyhYP9I>m)m_w{@4FS!?"
        "D<y6{2<byEK*i%a&@$f}0m@cmWRJmXJ5q<T)wPz3n}vYT1%&7Vm~s|ES|hV}S$L8YTQh%Bm2h0D9AQ51c^()t>5ly08dkBQ&I+y)"
        "i%3<KF6=;OBn44PXfLwqkqdFO3cp*-{VG`(kehHIJy}9!VF2(!?dmK>F9dlUsC~YVu!!T^3uoJ`AfN~#8f205;l?@(w1iEbLu$gR"
        "sEZs~lK{g^jIH4mak)du@_+gG`W0neG<_tGKOjxzaYM6i8OAv>rvmom5hO?PEht6A3g8tG2Z4~xKJ`qGw%!^#WE{OjQb(jCnJJCY"
        "h0iK0dbL{l)LGwjB;cE7%L-K2G<z6i<0EzZQl1t?B5hAAIIs@`d>Q9WLJtorx{CNm@Ef|6_S90qza(FG<j_XO?M!^zeAg(~F;#0G"
        "O;-(zyW*+L%K_>dt=D!}`&iba>s7N}?*U0pGHE(|mV)sa12e);2dIq10M;-IXMrT-P<~@15|igAq;xAlw%M}k_QaISxyo{9ijbLS"
        "AENl0>ctN1f?`oPGsrN$q9d;wGNTx_z0MxIdc!^99ExPlQPSk8R9$)kzKIl`Ko*rYg@i)^0ilU$)WQhv@H7yx_VvM9rXxb}%RGrx"
        "tcr$9uD}r`_$X<{K-yK;I7CP>A6ih8J0m!c9V?m@x*ZClT0U9>?EqTJ^wKvs)kJP=jw+J$O@hTgl31#pK=5UUQvNDiqd;~Nv}5tD"
        "y%hg@_RZOM-=3%6y<o56)%lCn>gB7mv(<Mml9!CF&c6MIr*Zn?yH^81#Ax4~N_CuB%sA^%esFOW+C6f4!xjGg=Hq}YGPZi}v|;_~"
        "S3B?c+6Y`}43^vNtWchUjRJzt2d~KMX7?oEYXTgs25qywm=W37H7jt}&#5#}m7yytm~LQLGJvr6sLb9;qw+$I<gFX?oTOi_-IeA|"
        "WPnGfq@#5J!<j)$GD3v4eJ=BMWKF=T2aHw#9NE9+0*oSnw55+WnU8KE>u2V2#m3{QblJ>>`iyfVS5|Iy2v*H>GLc9*H_^g18dP=="
        "*7kL&SkHXVlQI^qW41sJCS6J<jAZ6|VC7UGL?@+B$sPz8;+wjR%N)s8fqIgqsfy7emN1~N`U{>)*AQ6Z!L)$TY_$iS%yWD~J1;go"
        "@!r!SUo5^~oV}n<Ck|E`c)cgpj&lYG5|am7vc@9x&uSY63_6P0kFGF2$=JFm1?EY15U+f}`uA+7L+g>qyhGBqH);HFRSJP_m5vU?"
        "tY>+DWAd&>e^gNX|029^PYu+8@KRc`-<WygGHfgA3nOx#j^DSyG+{*b{Umx&0PMzzCIu;==@;wYE$2$r8w=-Dn}8viB%PB^Rd21e"
        "B-sySaQk@<(>i;=H5dB;6myH|P`4&LwgA@N<2fKDOnOM*mlcb_2hDI})CoIv5WrRBhxWn#@bKt^hMI$bWdd6qDqdF^YMz!aI?RLz"
        "IP;cae^#d`y>sw!URwHNu@ekUJ$YF9C-C}-&Z+14p68s#oZ~;oJDI1a)9y$u?f=ppm7u?IMx1K>l5C)}25wRIVzmF#MI~SySzJdX"
        "I@EYEaJEmX`90Y+X=Y8Vq34juKORv?Lb+*V_N@G~bZ^@6NIdGen(`os-y49eKnZm14a{ki*PY)f%eLh~FT-wnXlpQTog)-eVjZ@e"
        "<5<+CrsOeXnh-rO3|q5qf~$6`*5i=L!9l7F%A++T^Rhe7h6M{{9SjXk+*N}$(;mfO$OOlxYL8HN_14y$WeGKA&>50>@^ND}M;a(Q"
        "@=<CqjRFiIwHr}{lZas-p-JRVx)f_7ZK+x9u_>T;L@=Zbi~lM`m5KALyCM6s^leC5!;y9L>as@H+vjOcNU?fsnScx+qC;=qqM2mC"
        "dt-Z4+oabNw?%nh$fC%bpE`cDCov=y@+*W&DnG}{2rk$+S$=%XCsogQp!G}5Je19YUiPOJnH0DTJ3^`3z4h8NDfif<hXr-E%IS3k"
        "mQ_6$+nyE;s0?&pk*RS*!c{MnZhCc+K4@@-D4`Mff<s+;B?Oe@%#1KVk=3KraCpYVR`uR(gl{jrJCd&38GND2Frio7inx|rYX4CN"
        "Q04?3i?DrIhvGDVYR+A~ndGUJT7w0un?RdE95=}JL))YpW=nPs3oy*Nw9s|gn$)f_;c!%@>%)>DE@HL=Ez~yM=m<3#KHU$4m>l9T"
        "CC8m00ZPdW9Hq;0-C#!#O%IkV0V$`%(laAg4NdKs`+R%o6-QcX>^X*`COT|7+Fk88c(iZS0plv$)qaRB#dj5q6|tRGFYzo~Ix_VL"
        "_P}+5uFYmw9FuX^(h^G*jky*n+Yl*uYR~D+2F3mp*J@=;LL?x4D-soGtD93KtEGXdpzhy}X?o4YT~?I^=DfbW_F<mZ8NHDRC{Ta9"
        ";6%sK{lv6Y#<56yvdz3l%{2s`8!RB-SV#9(0I5n;v%v^G`6uF-4mZ0;*Jeb1Gl780*cS)0hsQqkyQbQeLe@#mw?#<9NzbaNi;L*s"
        "SOyXF%y%0Q)k@JRP^8|B>)sZ@KZkwMll)K@OWS_vC?DkTKYV|IL1DwC7vlSwx^(*@z8`ucoe-#(29L=JvO@RB8*duw@|8wIR&NXg"
        "?GQ!C)Y<~BqW>-&eyAvvBaBuef|^+)MB0`KMFd=kTI#wKYH~NxBAWN{^`A&qasq#0zuj@X40M4aps%x(-bU0f+z^^}``tA*BC5Mq"
        "V<i(n7(lKHbR4>VHe{8-Orwily<j4{9(E<jlzN~FZN>wqGAL#n!_2^>B?}Bc>`XU^6!^f7Sx!<db!66o_3fv6W*nva&da0v&fD<C"
        "S#p`~NI#<CoEp3Cr&54Fpmj!l8GVD95#R^!&9G5;N8;uNa<!PE_zZ8`3fXFe&cW2#snh4vAXWcx2L+CL??`tqR<O+yTmRk&ke2sj"
        "TY@!QUDhBJT8&I-eeZ~dVl#Vwa*L$Z`8ddU$J!jFW5g;g*2J*5PH#VrjHZpbkTNqe1SKglnkUQMD6&@46~Wz&<C`q?2X`qe8sn&z"
        "%NM(4M?&|ax80C%h*Po^SC%RVl|?!hw{ALGWeA6AU!Q3y*RK`_gC5jLT^ZyXrrNsl3WuU9)>wBBs|g9q(xu{-PA-p7RHawTNu_^|"
        "Qa(>1ds+PfMRXmLTM9^9WA<CSj>3D&r$D-UE(n>wwu5DLwLk80u%<sbW_4BhMvfU=6`ez;3DeQm#CZa74tUcA9Jab@Juy<2mbgJL"
        "5{p4-aHP-K8U$utah~iUn*fi9E*85)QLIOfk981@r%uR{yCqh8yZ7WPK`njyM>ZCJtJV$1bnod&kZ_rvV+tBYNf}3xjtrX>o=(v)"
        "?Cb|tt>l$JsCr4Zg%#B5E{}5)^9@_s#!4CKir^)1;M3cdbJgYA8%)EPA~5jh^H#IxiUnT1-|?mXNr2&`Du8m#s-F1$frA;91)kw?"
        "e-?AGl)uPe_n3#1(+{v>gm*OYLz>#Mn1SK|%sQ)GJOfE#pwM&`_)-zX3BIUl)ZPxDe(t6W5J^Gk-pB<^jI?Cw(+g!0IHrReMpc@*"
        "dX4m17^rD|E}Pb;xFtu36Qh%iK$XQfFpK2%^O>bQ(OjP4DKwNpX)3=t{r=knNXCI*oF5Wlc0U1rc6xex00AVEb7b8FmRAaeQ8gb`"
        "<asy9AAlWJ>p*vDU>}%1d8fwB5!IeRew=nsAb7GyP2f6K&L&ViNz~dVq=~KSHTBwX&t&H~{FAvkj_O!uj$?QdFJ<T0kdLySXbu``"
        "moBLuq;(WthbRs&GXg_}>bY!p9<`fdkEJAC+7Bl&N@I?yIB!etlQ!LrMi*S?Sa5v?ysF;oMWuPkOjCI}34E-~=pfOt-};en4gCYV"
        "K8m8N@)_1~_`7FM!l`nfxlw}q0|XXR$PPiP;@vZ$pKMA-F+Ir+9AxZ-3qAY({16sMTp?h(m#Y&_G%wE%VwqUUA<$!01hcg+2cbWR"
        "hl-_lYE1t}G;|88QWNTj?ByGFt<(L>FT;%3Dv<j>$olBtu|7J#JpTdz4UXyb8jdI>${+eEu@p|nezHjWpRc2j@BZuZtvoz+wS$HK"
        "l=Xb`1OEFLuZQQZ-PC!UwUID%o2er#S#q&{r)B2)jgpz86)yDt-8(z#S4G|%XQ-<tS!24pZA9&@xJM~McE<-lf;9M9qazpG52A*x"
        "1QV}n9y@~(QIFtv^^hOJE`t~Q=d=1ffF_PXpTzKZP(Gjdp#bMk11K}XAF?_)pLwa@RZ!JWnWikud2-=>CeX)U+aJSFKeu-(%Pp|V"
        "h$?4ML@SIUEWSjMUZj+itG%fC_7Z2K+6+cM{|it{0|XQR000O8001EXwfc{5-UR>v*$@B#8~^|Sb8=%ZVrgb?V_|G(Wo>O?FLPsI"
        "W@c|}WG--dl~-+V+cpsX9w7gL(7q&qU%GB<hZX^rG;WbLNweAAFbso1lP8-CC2AyP$1U>TcSlN=;@FAi0*y#feD3wRyCg}jc?OaR"
        "lRIxq;g!kROdE0Q7|$hxyFx3WJX5)cx#jV_DRarK0yjvK<nZ9|AhjlAl6x*RcMcp=Sz)Yiw&>8|!7rQj5q|L%@{@@z;rQ?%?6I>_"
        "X?aHAZw?O_{*-*fP^`Fcp8L|ZH%w^-v^I*;7RJ>(^zZD;#pU_*?DF%wPh)n%j^eonr`n)j^6tg=NslEcxrPi-T}ewcOseLCVoi&&"
        "y6a~~f@ZD&p;9HHLC8{K0phkWXeU-!vxJ=I0+M+DaqlVM^L@bHa^YQt{TrZe${V}xp;tH^?y(v7ViDjA3sOb*ZNA)(j+ifZV3K=="
        "!f4GAr^3Cl;LR=M>MMqc2abB$erl{x`8?u-OP*8khkanx)xnH4pvrTEnsEoiiwUd2Z45|6&bapo-qE%{(w&9o0v1L~up4MDdExQe"
        "3U*?-%1I}B>A<u4eSk%WTFnc_OR0QE<e!pWYyCik)_#=5`5;DXe_{dp=x`K)rGTMtF|D{(b25%M%vd#7IoHuZ%%mv_?dmI@lkL?$"
        "hqx8EO!-A+l}jTKV*JObA9h(m4)054;bE*j5o%X2UeirE`gjRb^jx?&q>&rO-3shFESN=uo2<VAHD7p^ft{nqE&h2X+O8;_T5Ag_"
        "SfFHV?^gKn_9;dbnxT%t*0GNvp`A&6#_xLU!>^|y9JpI&8F;G%CHp_o7*E(9h1!*Gl@O54KuV0gIu^&9+gHY7ZgYzyO@ZJRNV^#N"
        "yCFiBwbb1JZ1(C{M<-CkXq5va!p>%T9Tb~!al7JHK5OG$X;q<N8hM40`XH;%s&uZs%8P=M4Xta~2gowcG+$%3b;_+P%B_9y;ZKG7"
        "zSO$99@WPew(>ZW5S9<(svGpH{m_Egal@NNR`t|@PKUx8EODQH4`#L-GH2nNPSBH`b;4J?H8isjhfAoV=qBW<$<ej0EWj?6qe>s-"
        "z|XD0W8H9H?ofGy8y=u37baC2ac)NMC(IOHWklXV>-rO>jVU@p|L_!a5y0@N0zVbLM7qi#Bg+ybRx)(_8n?JZ1|_p~3$aOza8ke^"
        "%TD)+AeHlq3T6}PpQ~>IxUFj@DoveHC-93-6sGcOcQ^-h0N=%56OPgD`zaO^Hpl<unnlWu34j{BMFFW<LwQ3evI272?^;P-Ka{B*"
        "oZD5diTZ6Z$JpBumxj{ckBuF3IWOG8cxpninT&<_-xA3XsXB+J==S`@&->dOT;BPb30wBjaHpu+RVUa=VI5jM*W8RKM4$B)EV(X&"
        "hpLh{E!KYV*k2eseU;Zfx(R2P6UKsli>!gt&G7C;#_cVeV>B|nSIO9g{x8Q(w%U_@j@tx!<GiU$A9X8Np%IjQVHVW)xA<+0G@x>!"
        "OBtH0NV}1hsKuia)Uj(h-O7sHx!F3vF)#d($mUU67^A7II+7{WJsN33Sy+NLNS6KZab&7a(v2cJ?<0&>0ALMAotwTMoqCR!Aw{6s"
        "xU#G{+lehk#LpdE*%G0d7Tlv$#9lANSbL=SIR1QniFM32b^WRI8%JAgnV^}FT8ym~cYViGXb(`1F9H3=V;h3=VVvNmXMavF&T)?W"
        "7ADjIjfdfhrh@7TlczoO-N!jF&X@0F8+~m&8eOhO)*5?+tBb~9VehyaKXJG{%|nNfN2m6fG@^_r>nRw=u^c&yvy{&)4TSj0l|rD0"
        "(^4~A=Jk+FS4v3PbV?o4bb3U6K`+WZtwvo$0MC^U3)D18YGLRXRXkyBbNp3*|95kaE*{kTc)otzI#DYV>Enc@3GTRIwBLs>7w4lv"
        "Ccpbue?8D<1@`FpK55-Ujm6RNV53v>ihgOb`TG7%?Y3%6zh}w?)732)$N~En)xT%S_C?o3$<9OJ@S?}!w}rxh`43P_0|XQR000O8"
        "001EX+fWH2jsO4vy#N3JB>(^bb8=%ZVrgb?V_|G(Wo>O?FLZKYX>Mt5XD?r0X>MtBUtcb8d2P)x4#F@Dh2gzY?_hbS7eL(*7k~|+"
        "$aRxqNaD&#k+?kpiH+?IpZ<r42QklecRHRya`hI7ec40UVuWfWS@HtrQ(FZHX5}sU2IntdjH+mcDJnk(MC`l13#JJs3Ca`~pArqf"
        "(4f=3y~)!TkhH2*SgG>Ov1uu;bxqgTr{NsM=p2l2!9N7AefI`XO9KQH000080000X0CY7*IqU@h0H6^704M+e0CRF<FJftCZ)0I>"
        "XJu_|VJ~!YVQFq@Zf7rJb#h^JWn^D-W^^uad5u>~kJ~m7zWZ0O$|3f`ak52=0#1Eu8z6^1fEK+thCoZBH4%vvNXom;V*h(*NKvFD"
        "dr<+q7U#v`%<!90mSw+lTb!U|Rae*VTBT}c)mM<>8@ED9#%0M0)wDHOcn7U2PVDbb|FBAF{OdGpxzOgleN^no4b(z{*<@L^Scs-o"
        "+Ome*)1uO<K}mb6#gX-lzlh;EM^UMQ*ALJ%d{c7EH>#@0RBkF8<ZH#tyy!HyQ0C<4ni<QFHRP3$51?Buq`hTtbfnba8MLSbTI?l2"
        "Uol}d7ZNwQ=2j(ao{(SDZdEOcOHeG7ti(g7A@}sWQ6=D^N~m~O+dQb{tyYz&!MtW#V`hx7N?%gn!kO}ft)Wo5Tr3v24{MRZ(Ss1y"
        "U~g$!DAtUkoR<PsNOe8f(|N(MW>(t;!w0kdi;|{mfbqI!=;b$%?DMaN3CVCT&>xvp%y29sw7q5;-Whzg$Ycr=P}S;@m0Gne9fgXA"
        "%xd*$ybEkMPyTZ3+sdibq7l}hI5GDbom$)Mu+N<1hTgnLc5w0~#sb}=m=}4&rKrHzu-m*cTvRrL5*6@r8WQga$X*6tQo~Efb*{d)"
        "#i0Jm7W9aUrSbzEY6EQfhmx>n^xtdv|EjZX=YsV7;s|t;T{_lGSZG|<DE1q-TJ}j!RQFE$fEM<Lfk#$Qi<F&S)S?G#-NMs20)uAu"
        "=FJ=SSsJUmg6`)R`HYtKZ$%`%J-mEDIthNltFUEWf)K^A{PGCs1t1|CZL%7zLNrTs=Jx%VyZL-~hnuWf)}JNYhEu()XTt9K%%~x)"
        "Ge#(>TN=5m+MTc6eb!U6!%a&Q1;XgOXJv<((ycNYyr6OlwgSpWd+S%CJuaCIZ(ES%P|@Y6l=oCP8_GJD1>~Z1MMXphBqK`)_FcT?"
        "lrR*h`|Y{o;zLYB|NrH62ft~p^fLS8A*xU(LP=4Aw8CCkapKrWN>+B*N*+dSaFcmgle_#pY?7g$*ygYzvH(%kOdU=U9OvizECwla"
        "<fLbzo<B)SOQM>=Q;R^+fD?6@lDvULpa->WpQMBnqG^Ur(r_h$5oEYA;EX*Fg|;2x7~X-9`|SuhU#5rb0eCyX<15>ilYq=S`}jsa"
        "U%7Ia`7PVe81yWk(z;r+r%>w@2U>=EgAK6~Pl0N=egI3W)m5`ppHBni%|zxm?Hbnhg&p8_r1LVYRz7Giqbi(~3>M{;xCLQvS%d9#"
        "yiJ+w_?VJNDd9}(pw>?cH0Ffj#*+5g@Dq!Bb#~T$S3R$jR<j<t>{oaG$BmO_ckfq68bou0)etDWD7vPr-3{FR5DY!##zJd$uD^LZ"
        "XI|%k-So2i@xupCGyDuwng*Iua+;$OZJe(jkdqoT0Ojufr}ZL^IDblDx|8qwDUBu_?>;+T!p!y&Aq0xAt-`po^Y_ejEmgc>c!P1o"
        "!->axFw8iJdcBdK**%RzF$RH>#vst7vOeHpfgz8WH^;lr5(6Y=bNsO@vAhGcwKbO#v)i>eF#`<<y=!-FhTjw*>SYab>2_YR_iUMX"
        "dpB~lpHch<G>=Ve<GF`CGGW}1;@iwX4)N)2;K~2b2~}$m0ujcT-^wFmIAX;cQC~(=NPx0j$@&uK26B)R9dIHzIT^7sb7>l|S`=AS"
        "aR_Y%R$q@XVT(ruUgI@09yQbHtp7yV-Tn$#uJ-;I9$l_~MJ)R&^|Vzl+2Fh+j-sYsH|k$p7bk?D0-*t@i%FC_!)V^V(#J+^DLMM2"
        "IeBkU#J?mQ`ZAl~{taY%y?aKb8FSQYhETewxVA(uiAXGH<i<rJkybg%q{~E#@PoP-{-n4~x4)+QF!^YV<(MEUJqfDPGaxHH?%Tja"
        ">?%@6S&rn0n*KgTPLIL<Cz2B|z7~fOGG7nEB;31cuoKXZuaTYLcYKZP6v~tLC@`RUcu7nXy2sqO$WkeWI@*n!)hfvymRJvEZo+d>"
        "3#jHA@C4~bs-X+yLpGes81niC?R6Ia0Z>Z=1QY-O00;m803iT{t-9qN0RRBy0RR9g0001Ua$_%IX=ZO@VQgn*ZEayMbaG*7ZfS03"
        "FJx(RbZKmCVRUJ4ZZ2?nZIQ7~12GJS_ml1&tW2#0R|cj-VuFqg&>f{Y=Nd@j!*L|~_V{`Q6{(qQ$MOI9k1=L<ci#g|5-8gU(F!u)"
        "_k<i0W_f^20wGPrf&&l(<5vx=@%{*d22S8@yX%ZGZPPYOCkU}S%(-Vuun(S`jU%RHyW*_(6o>Or*rJun_(JlEb~rvW6VET(W<akB"
        "MM6|6#e0ub%Bn~m6ikybNa+`;Zil%L-+-WX2jr`_Yq)*<SLoXY^eFna&J6v$cnmHG*ciq<8?Z^Xe3p%1QcF&xVDs>+1CG*EMQhGc"
        "%@XpsI5E#<8wy_CZ?!a>#MOUu1Lg^R$T|l;s&4Fq8hp9?#h1VyJt>Cz7jZ7A=pRqW9vQR4?hLVRn;%e10|XQR000O8001EX-ZEcn"
        "V+sHO9wYz&DF6Tfb8=%ZVrgb?V_|G(Wo>O?FLZKYX>Mt5XD@AKZEtdUUvP47Vr4FHd9@l{bKJJ^-823VDE&}UIWw}G>4TEXjFQ$*"
        "apJMn<iVri;7Htw6_20+Q1;p5|K8mNNPr;kB&981d?a=k`~4*+N%C&bn}*5G)cg7~H??fP{r&CbYt|{b5i8b8!_1Aq*Qzh^n7w;-"
        "#f;*0D^&jU$<rsVWhL|lQ?hTX0<h|qNyX%yPz~>jo#Ne%&e;_ZX}B>$F(nLOYlau^0lCqCWgKYV(e+QCFpI-BVk?zk8(!Xi<f>Y+"
        "O>bBwg(j}bhSz&m%C_v460KoB|LY%@>{h5&G)#OFWslT=!z4+N&3fNS1;W&O5xz=Q-b~-}wq+ddXlI}EZQqg(c!LnHkj~R5z=+oD"
        "4eIL``S$V`Pz+ZoK!alXMr0Qt5j>UHvZAQl+7v~qMY9EYxE2?1QEqWoO!4;ig4H0*n!Wh@fQgUpJE77nA1P%1tlknL`(e$_0~CC~"
        "Oj@wtdDDx3DkW8#Ad41!-)jRpW}VbEC|~4BKoc3vkw4%B|5-T6mH5!tN@%okRiv+YqeV=DZd%iY(TOg3Q@1-fe3_p!_6*#iyW#lC"
        "&X(XCsd#Y-=b*1QT=BgCUrMLqT0@C_XAUWnuqw{$w%7`8df;^hcDSocv8IEMqTbz@+<t;1bLhm{64^GAo4>sb<cUQihJJboL!X6`"
        "I;9(BZvtK{kN*pffCqGpyp;+9+B%q1*@|gXt<UpwMD@N1v4v6PT=5QFM28*+acR&Tukyxv^!UTVT7J@O6`E)A_EW$g#C)$>?>vr3"
        "#=fx^<wg#4N#+ckAaoNOYRET4{I=eK{t8M_1*IZ@NGX-9*x3p^x~a><1<0*HfCQOIX2Hdt|H!JkG)`m)B8x`J1#;Mz0WaD1O_O{L"
        "R5OpNE62n@V~8yzf!PpAjPXT`#?j*t*K2p5gXI<PjHrO+_t!C_>p`}FHLxmc!4+3{3uGUZq^Y&B+R5>#D(sO?vn)y%K)cqEnhkGD"
        "ky@~oPf!`-ZAIjBrMsp!X_AA2US#i|zqlryP08Vtd_X(>J6SCp-5x)>JANb=kG@{aK|79xp|JXZ+M{ZyAehk>M3`f1C{uvrG`AC`"
        "(EY~hBn>EM0qNDO3=BF%=gSr@arVu5*NL`DiJ*si#xC_?4~a$9CFI9?*Y2^>)li+pM+h~^YGZv(i9L9rt;>7^b~jSyo)BtG*z&z*"
        "yHGIjZ_$?UOBEgnH1ex##k9r*p5Y-w?}#kmfns03QvE>NGYlTy4Ox`j*uQ~b)(GP#@LdXo;eC38j}&TOhAU1+@UJr9#oxm|3TQzf"
        "6M-aPrXeg{It%<gu|b(!xabVrGeP6S0oB5V@MX{e2~v27WIaO-!3y}!S3;nX<-)TJkh<TeUN>#O7Y!(ErqmHTTc)`=7`oVqR!e1H"
        "(;3oNpPQ$E^r9)zLe_QK*FvBVLU0W&jFATVDd=77D{^U3iA77~B8-VQ-Z~J5tZ(ZNy?_o9imaw)wT3eeMe}<mH|T<_t{`k`bAT9v"
        "n5jh7cY5e~K@|{LP?B^$b@$3ikPQ_9S&w`iKjpltQV$&kxs~plwUcBt#@X-zTnmG`(7;MMbuYs8+8r5o3Lf8M9S`WCH9S{TW~UXy"
        "s{)(%k?a%*V&8}fKu(`>TeriG6v62$sqF~RG^Nn_$~8<^R+Sbu^BUN7eX$pNsSc^_kQS=ArC^T-s2p1g_7@1avacrf(y0nbM6DZ4"
        "C$a?2E33eD6rUDZ3w3u2VylEWeoFQ|e%uLuTReabf;-KNQzf#q8UxO13N@00QVXnbb-R^=(k8wXDJbS@N($`b`5wrvLZwf5og)tb"
        "3%G$sNg96=YLk8kXN#%~@8zG02Qa>*>18TMBnW&N%Hy_feE$lSOZJ^D${Gh3Q{Lc=aEAT6==Ay}^>kvLOh;>|)eTf%A^}-XtY3)d"
        "-(@}?FnL*`*oZcwbys8<5BTbC${58s+{AuhFHY9VcYW722L>)?*#x~Pdj~a$+#pBLlpsL3Ss?Ob4F%p1$M@U81_9gdNX*j>I4&W^"
        "uKV;VVz_@Sg;w(#JCOICt&%5Nj;Pfjcq{vWqp#x&>WZjCk?3q0rW+Aq#>99%y-zzCaLqJE$OfxUZX1n@b#{7^-K)n2+e?@^>b4Tt"
        "m14`;f-P~ZUTTuCn0vKahsP#cd~#!let&*FQDo-w^C%r}4{7H<BJ1wgf^JXQ=d@^-<GRT^iqO;=fTvR9pn;UTdz8nQLo{KSZZPiY"
        "G{3C){*ROl0@0|}lYm?y%Yv6TYr7#i?Ev<IYu^h+@&_VX1&~2fA5)Hfs(9;#iWlT7eQ3)YC0pz~CP_bn!(?U(N`5ZAVNKUl9Jj4o"
        "FBECl`5CV4D2qF|9CF6CC?QgDYvMo}=-`A7hF#tAPTxo~`PP@E(D;S)Im{7<XbQ$$gKC4WdnZy#Z1IG0qV-vrCVkt3`y7gX46nk-"
        "0;6LcOf|zZq&T!gl+Kne=uOvKQOUG6GiTllTvb3$Ki3PTLgSohEF;%*`fjZX<n2v4s@p1o-bn+NhXrgGrGx_mC^i^`Yd^G(daUB%"
        "Ig)9huc#=sX<JgbU3A_7a7C0SOUdCGy<!-fm+$<Npw)34f`6c(fCKIXye5!G3%CCAMfB`!?zms)gcsPEYvG6zIC!@*Y)~P}jcQy5"
        "b~%3|Ws}-ito0Ume1dE++uh8y{T5;iB5fRSTZ?{NulUg6uyRjN0Etoigi;9uDK5=&xM80ztvb-$2)c4gS`E%N1;t@_L9{;=1V>cl"
        "+<nc$Fgt36Fjxl1$1U3t;{@Q3w$bZa!$h?Ewtbkbf~y{BvJmrr%i)98vsm*S>I&Z^m7;Cp+7^J^Zz#;KHPmKGSY3{<QFCP5F**pM"
        "X*qS_si)}7hyB!&33pkJN0}?tF0L)^=>&J5IY%;!HXYhLNVCM&F6SF)YjRxYWb8+F{@*f$Es$^EygqP^f<<kSt8m;G-I9NyZ;ZsE"
        "1xcK$k0xcm-5`J?2|1g4-~X+GKcJby=b;T1?Lnd@5}#-vbB|baz(?$529$euGfw;m?Pu@_f)1L+zQN=g{V@P7ZF*cMCT}j^y^Htv"
        "3FTtj5CTjvr!D~k1pX5rQI`jeDs4?g<Gf7AV56dtnRy>SZP#avdPj8&Hu{4Vtz9KHxak+jY^fRkjz#;-QP=wGOrFs30b<;`o>=!C"
        "CxIa|B{SX0#_Z6>EMIi<*)2Yu#}dy&jdak3I(Wn!7qwj{7+Dmud-U8GUvSR~ynECPKclWiDcg#u&Icj(d>E8sG+V(jHt0uuZ09dH"
        "wm;%yzBT0U^PD<kZ%El_-{E%*6zm2H`!{O@#{?Qf`v0L(pj#F8e`x4}{{M!k0E^-*Knu+8XFVi6Tie=P<7Q>BSkuH);KPc;pBU}p"
        "WLx6QAFa3TU_%l_7d~S@1GB)D{MhQjLveSu=Le7q*JlDZ5_?8=W550O`g<TZ>I3?nE#E-@gp102O8fNEV0dg4oD}yzP)h>@6aWAK"
        "2mk;8Apja<yqj17005K%001Wd0047xV=rQ9W^ZF*Y-eR{ZDB8Ta$#w1X>MmPZ(?d?V{~bDWpgfYd2La#Zi6ro-I4f*t0%hxi=-P>"
        "Iy6eAj!~8fMJ||N$()gGr11BJn503iH{W~r?%q8@$RS(hAhf!mte_LVY*K@IfbEXM)@UufvEUk2dj|$f`3~AR2W{^>pz#3$aS;kR"
        "Ata9CNVT1@9>&J9S^hX8uX`-LGAJ|%2m7fZjw)G);pfth@@I+Onr@(R1mn}@qP)vzje|sIYyw(*OZ}%h8ZlO$6;|?^gF+7*NLN`F"
        "M-0-B1vwAus-6dMtW*mx8+e1!bC7@3)>F;*otAWFyGC^Kl&=dm;WHGPZV;RGr)Pz75{^Rma%Zivl<cmZEd6-%pa!SU<SRKAAg~(a"
        "uu@dRBbK0gqZFL5$J*2Vzm3s!o;&UF?4NOg17(Xvr0*@>4ZFUj3|=7%gc;oP|3df;g4fyzPZ_uJW|XF@1Qth0R|UHV^u%Jzcs9?U"
        "Fl6-h15ir?1QY-O00;m803iS^;~(v?5C8y|KL7wI0001Ua$_%IX=ZO@VQgn*ZEayMbaG*7ZfS03FK=dMY-w&~Uvp-3E^v9ZTYGQZ"
        "xDo$9K)wSjpg74{%a7h&3pZOJO&qkq#dq-KS`^!bz?HbWs`aWKw)cX3_s$GSiIlkOb#fJO7mJVKaQK@iwVvnwwd{&SB<zRs_6_^_"
        "{l~L-E8@GVOpBInMOKNrVe7JH?N+dyx6qrg4`s$ztSDQtD$BbjeEH(#i#G|cT2Y^tMYdxDjfU~2>2g8ytF$O!dR2?I<|+Q!lwBPQ"
        "7*q?!?|BMzGXeB&N_P9hjOAq_GFEk~ERA`amc@+4(8R=}h&y~|$eGBivgR4%Z7T|DjX7`tWIWG9-clUU!qjTlzifH4&C*r>{-35S"
        "`j?s)NtwTVv98OURlLQKO!fT)ud1i((nL4o-`^UGwPhXaaSdM!V67=ZmZ}?@qOuajbVx{3hk^GZTBn&n+B9^PuFIHb8<FQcOnA$~"
        "xXdyRv-F%>QSr2Xt6E*tNd9TD39AxByVDc>moFGRt1iuwNRNp&b=g(TO!k#ko95}SqL$53&Wm&{npSmp6>uaXNjAy^uZctrjYuKD"
        "UA%meh&97KMM=7Wxju?GxRMlP6PxQ33Y#sz{O)@{klhbyyQMRblR}iC@6{_W0Hm>|tp(2&C43ON@wO}OSX!_Y0P1HvUnTrX4GrOc"
        "iGOi^`7JwP_)jonE6)oI5(87A+?`JZ6+eLGt}gU|t$0*hGIknQ9SBnh|DI=^(4^BfOPjQ4T3*DeYbH?>ek356Jg8M@RG5es@a+d7"
        "5Y}RGQcKdfbx`?qCw5l@S0I%<AVER<426`Vu;g0!*7(8a1FK|d<633mSiqvaJS2y%%7lX`wfj#|wvUFIhSMO0YSwKu7%2sNvNmIw"
        "t9M|p73%N0D*(8<CoXe9V!BGRwB4z;6W!M@5do86KIfjPhLJX*#LxYOhZ>?bU6&;)0U=#27ywwCGP@W0&~Z$d-$55BLk+gCnSZQN"
        "eBu$#Br5OdRSjOJ1=z^BBz(r!-oV`HC)(cPn@v`(cosFSsJ>ZVoqwNv_WDDj`5{pTK$u6v(?+oOpw8FSd0p3K?XSJJgCmi%{tPFw"
        "?%=F9th^WXLk)*^#Xb??&tA|AdLsrleJ_NqwJFa|J@!gkSwKGMnK@nwK{deyo`W?7YddG321W~Ly<*i!f53Wz)7W%n*JuLTaNvX)"
        "s}NU&jSw4>XrOOS01;JiK>zJhI3+NSJdu4$b$UDQ57TF8gnYT<Gv;yWLYvTV242u_YL}Hfv3KP4F!c<D?i)io)(Fjj9|SXFUT@oJ"
        "gm-1(TaG|W6xGEBI;B<Gh!)@F6yG9O?b>ZwsL|4S;&8{vxJ$T26&q~Q5zjI|KzSM?(kANT1LzbPKyI;|St6FhQP94U){OPSl@UiR"
        "mk6s?;|)%NqZ1toR;nvXb*#eXZVY|VGF(`VOKb0@r9jdS%Q2h|?leNcHAot~vq%i+1oqZJSQAFse9jJ;n@i77Vl9S+H*)Zew*=Ab"
        "^ZEwXh&MGy$##IY9v@<u<pm+>e$<M`*2k7G>GG=aH51^3ASNx^`6YN`Q9#HCP@Q{SyFUF>e{Q_K6SYgh&A>u4fISZ7M$Pu!dN*@G"
        "W;WS~bqK?DZ?E6H`_~hDCH{}uDV(h|a=rBrrL||jQ)Uu&Bdznk5ff|u5A=!BP@8tT<u$4;6HUx32+zUsKoC)j7*uN)S|%FSLJ9h`"
        "u<dZssOe_yb%8UW@db|8fc0UYi@<5Xjx;d&|Ko;jR>nA_vqhuR_`Y&pHV!b-lrWeabT2>h9%SUd0+oYJ0C^~&1q2069{|IPoo^_F"
        "p)-nkQW=~j2`un^X=#ooC&^4hf;}KhAqgt*1T)wcqxc;9AW6hXElnuTV{5SWA_)g=mmAz8=dpQtTGSw%eiA_5*9M<<Ji*JMt=Wl!"
        "xlv_?^R%JF`HGQo!~|F-T{>7`_m>zNQJQ>zAu}9GmuqpKLRivg+MzcrHr>Z!<Y1Px6|?Es6B3V*_NO@*kTcY(v$nhw1?uEkubn`+"
        "Sf`tQbD?3azRnX(PG;he=!~B0kwyvK!Tk@rq{)EM1olq8_)MMMWZ!nssY1<Tqew^aXejZ-_e-tRr#_ArmPcEAJ#k>3m~Fu3uh$?z"
        "ZxVaT5DQ;=-E-KZoL5m+%8Je0gSZsh9z?p?woQcfAu}!cYP?{)m#9+)4i&FP9OEq)o(TnSX@PAokW9jdA!kF&4M0P4pM>Jixbhf<"
        "`cxQIbK<Th2QT5+pOo5@ZQSw6L?Jh%WC2p3Jz=||W%Sr{5yUii)-MZK{xz*E5zY2&oRS>KK+Hq`GBFG&jmr*8izAp40!;{vvK)2p"
        "q|LMy&@33P!ggwgfL*`Kes4L}LmDS#m!h}i#YXsM@DT*))Bv&UoV}jwbzVzZ%h?At^zQ}vK|?J-`sgidWr!rj(2fr$nTKs8tgd4X"
        "1dg6sI`U6_{nE#1Pul`_20H`D!ofY1_1)Yf1CfAZi-j}|RJTKiEo$2fVDKW_!#lA9D@j7fR>KrN$E+*vu*@~Vj>frRtcO)}6>PK^"
        "W~fw;Qe0vd+kiR>pJFHZzJ)+qu`UN!A+?Q(Ik~rrC^XH8CdkQrHa8$62Qtw`NnT{a^K<uzPZeV2JTkW_&sOd;7zOU+9wMXk;4;a9"
        "m9krvg+YD$5`-SEN(+rHE%ZZ;0>pAqi@v+BIxKC7p+-6e$<rE&4-&2sS)GZ54npqjFYn$cv`eF<aMxJCGnQ&5l@J$yqjU>~4w;s~"
        "dM8dsSDBR!q?(joK~zUp1Pt=YjI9tuGFMXk>3#K4=Ujy=+KL*vT8NAI{Ta4A!_qYbi8$*#JfE?1)mNKA=aKk?4b|jgq-_Oicut!A"
        "z+~P9SLy{m>{}2&=qW7Z0#|CGR|dUh5VobS1TrOEc5T(QS~5P(a}486BY?0R9pV{a(t{r1rYyvCJRlJ}A;8IeDW0@}1R6BbH)sT|"
        "%?=2>1$05ot}u9m;W<*kOT6S1$S|Z~!A?ICm~(~+x!hBj@s-FLRD;ut^Yb8t+{6zYQ|av4tGT(?p}vYH1%lfXxq@ppLFU!m96*(W"
        "%!G{rLtmqFP;!)<By`ZN{Y{IZsUQ`?%O#k}8py}$dl;K28ohw@U5STVqG~1X^h^zdoO_rW$u=6KIyKaw&BE?y^Zu)i%Jn*f;2Blb"
        "Q3|0Q>*|KMfA;M|&L7F)H*+{c)h4Pk-a><V9gyEQ@i^}CE|bCgx2n;)$eUk^)9+Nr1Fv%kh2=83G0xHoLiReJUxfX_U@;<bFY4X="
        "QjepW__$4oWPb5Sy;;hNBMf=xs5>)lqsk3*HD()zV0}n#cJxu8jFoj)e-EebYe9GLXe19COnxJ>4^)o9-Vn}vEuDbq?;4b~1zAOn"
        "fSfo{7I0E{a`XatjVL__boVDVQwL=mYV&AjhD2wEBm?%ET?|ygoLXmxap?Im1sffOYUyF%)xy7x8F!NI;Q$@+0!*o*qX9t6VT*|q"
        ";PCw5+Bbj!AXkXIYIkyhGo%dOpQ+dw7N~%mGf8*QTF0%K#jOD-B1czf`cjKpFn6L>ye+0%rKc~uMm3;Yvuw(EhJn)#D<&=M1V46Y"
        "(>0IrLM+~vCD1Fx1H1j@Cc1g^;TolNd;Rk-AFiYKw?BSx)O?<G?Pt{`7P2B?)FPg!LM$!e4TBFKU+F3>48U>Z`v=qgi{TP-q|4}5"
        "QJoEThICPfaMTjMNqGKG>+UV9=ay>d8by?dnD6Ek#sGf;eA)ax&l)XJyn8Yh>?;q>uO@WxHDWwCmSLSkAIBO=*YpVX{RloMC;P6Q"
        "2)ZLa>6WV6flC(^{^5`JF`hpu)?Mu{o2b%?LE^Kqjc0=7((4^12HSJcsmIozf;>>syip5aRw|GFnsylkcGhuA|M=*jARyXy;Jby&"
        "=Z*Tq6@7I<6@2@4@EpSeH0tVaS#P+~<p^DGke%BUuS-z;i{(%WG>n&708?d;gI6s2G2@E~0!Qt}a-*I@5E|q+o;10IIdr!=m~G3k"
        "n}M_;gT4Y`4R+kX*(@3Nan{6x*K!jc`^C}Y$h!F9yFXie$6R(A5%9!_l3`bdI^&Ib_=LTo#AVlqY(U8?_0x+%?4%}PyQ>7%V=?_S"
        ">#j!93S=N{tpvY|a*+dWDSd71JSr<U^!>x|ZHGoV2`f`_ec0j{C&RCYh?YFH8YGnpyJgHYsv_CdJoXn^hqH@*<IHMA84{EWd%!?g"
        "j{&-q<@K`yOMT~DL9PFk1=7#d5<&KAGcuT_(H5QS_hRXzLfI**V}f6TW*XOJL{X|Rk%$k*8q_}|^JWO^sHc6=Y3&g)s8t#hPxkc1"
        "c#*5gvQ|mpu;5-v4%K2KDO<t`c~yxb@h5eO1s(PR&A_9)VWmImi$zIx8CcaKNHQaxv}0!eWa3UlWnBa#Ka(r_;-F8aCH*H$T{n&t"
        "8kZP?ng`fI`C!!@hDss)r|LjxK-x0GpnrT(kPBW=A<N92G`!?F#Z@=i*ob#s73wGKE%Y2w<athk=PQakhYLG>X67vHOp7|S>0=5z"
        "CSUA?`I6L`2@D9MRy@9g>?`rdnD3A4$+D1>vP})Y|CU5lDr^ScJV}M}e^rRDLv~}NJ(Sf@uW?4p3G4@QDwoCLtLxM@K}cVyNv+!8"
        "YW-Yyao9e=4{3wnK%N!TlkDkO$4QB_P7aD9?AkPILdvmm0wvm!mD=?A_x?2D04#Ch1doTYs2L|-E#~RVZ{XfmjCBq9FMj7Sr@9B0"
        "iJ0$n`mfkU00=Sb2b8E<1qF&-E==2SRIW&zl=_z6G+WjUrb9ESW8r1WcUu*flYV!@yiv|*3g*xk^TWpsGiONTG^*I;f4R9ujq7nZ"
        "vu)}q>6o0A6(`l@!!-T$j{8=UhU#p|4se_qYow~Qz44ihqA%M<m0D?W#zU<mpaQP@bjS0w&`9xM9;w2}=(%&<Bcg45%xj`u5pU}f"
        "KgyXpF%bO1lQEnuIub}7Q5-w9LOqKi2~T&nr4|XRvhqjb8V9V%ij8SvhW>KJ={BH#ZxlIEEaxL8sQ1V~wPyz>5fgiF%COjOmd2--"
        "Wpiib(33wx@FwfiO^r&*ujVe=&jt|Dpfn8}7=E9FA8({9$>jhp%(7G-*N@XE15<o?a`K1si6zh#i=_^8VX^Vmv`zulk9N4{NO+yU"
        ";*smn?{z4nHj`;w2fT#$?_%^3jJ_lo(b3H#2KPg7sI9voXeHZ61JTIdwY;WIX8$i$#*eG2qn~}%9qat|^kJgDr|VD&Xu8W|Dn9xd"
        "iE6LpOzm*%*z{~hrK;nrC93p$q`etb__s7OZo_b_c{%WzkV_5*%%2bOC*ge@w3VF3LC{R%as<I+(6$#@PTs#9VEXcSpz7fk=w~mJ"
        "fJj|7K7g(5ZTr<ecWvKHjzbXLJw*yooHp*8J{5ueLzwCG&*slp@Q^y3;=TRx!%sh4f4qL@efh6uo{R#GV)@t?S))XjM3737a4d0H"
        "ztI_xYwg+GtY?%>Y0=s@S^{Tb)wqu)5+g6X+O?vwe-m}c3qS4o?bO4)JeF<i`zKe|!+x{H*IhvYG#vK|{QnkzU=Y<^0US`4>OPO?"
        "xSwQw#Rb;0F0T&$@JR{Bk%3NYL}L*_UhqFqO9KQH000080000X0JJtA<`W120A?Hj03!eZ0CRF<FJftCZ)0I>XJu_|VJ~!YVQFq@"
        "Zf7rWZ)|B}c`k5ywHZxs<2LplApe0?FVfVbX<D?0?D#N?O@RT@HfU#d4~8Ky7M-voQ7TC#lgawO?|Y<3k+f{5Kv$Po;`{#QOO|C<"
        "pZkVyxF%cL^5%*39lSmfDr!<Q&1B03SDJ4~*K6MLFHDj<*$urS+fEYEX?ELnpUCe&eJHamJ3l)=<89wbO*UQs<X>8<?}Ibl$<2K!"
        "1fhxufd^aJwd9s>yA5sbSliN4OA4&TUD>j>lTTGIyIV#^0yp?_-Tn6NnhQqd0=S7SzZ)d8aFxSG+UI9A+k!S*IyBm!AmElaOy$Mp"
        "`5A#{)3t3U$dWwUcYLx3CKc~Qc1bd-H4_?Ms-}~)vX-hIF0w@cFRAzhz!yJ;A8FIyQ~3J*cj4EqWb6y89y?hpIQ$?7HgnuJY~~Q?"
        "Dg2)O(Q%=j1Uvi3X@;tdpU_#Ohq)kmrk>iCX~{R)0>qo%oGtJf{R~6|ZP_Jx&Ke0Uzhkd|0HjAM+o7+NW<5&(!_Ec-Kw1t=Y`U6B"
        "q+c`pv)E0<`PRv^Tt*TE0!%`zz=Zrxv|#^ijS?LfWg_8Bv9iq8v3K!#^$H`ktQqU^M{c8_Sj147#Ruy4M|y0cs+xqnJfK~QXVn96"
        "`y)6`e7SN~tU3X5B#N9?Vp1R`6oTw4@e^%S6n9?UXYbXMfw7CDQ!+hB;T4-Q^Z_{6RI?yOO^W6A&w{kTuUY=m3FgAo8F?&Q<sH+x"
        ")3mT4Q=mby!q9v^82pRGHoG7eK_tb<ziBhDD=9mfXCK8U(LD;{q<uY>PRp$f_=m!|dQy<DmPbkofm{mFY`$A0oAO6x`xV3|299_Z"
        "1nZ@%?++J*E24%T%PFf5NgI`@0#VS2Sd3BugK8DQ9ToFGK=;kf^_90F5scM}P@-5*CBy=#;liSJ`~q3^Q_u2H3{!p!F990B3%atW"
        "Yx0&{92LtL3Smh*C{3X4P#~eW<_}ohG`nLmlp>#q8K#O0rK#A&qS+=;EJzL{7G&FWR2RjBYTK!4l_ghOu!nX8RwME{22T#j+a>wd"
        "0u-n9Ge}-R%g(fYr>WPFeeT%??Y|`Xl=i!%gh!b}7}O2Ow3#NbE69ICi#-RWJ*64srlqLBkfT(S`SD|D=&?pEp!pRhW4r=M%%=6u"
        "00*MYOY#y6-`Fg(lDzR8NZmOfZ|>7V;o9c{71EFl3agd#@u<|g1UI$Q%z{q?^N%gM;30==HH5lcERt_geGUQ&ESW?v5{`m(K1Lg9"
        "4jO3!FzMfWw%KOyg-L00{hQI*Oc%Q{5yfP-6*bM~QP;PBh1ZBP!w-0wzT%i*d%286ITYjxcSookgFYmPVn>HKef|MKiFJ9N@!{b5"
        "mlfhQ3gPOxWPc7^GAy09&?y=ii${i`T@5|cej^1TETBus+#o=zPFZVuEeaSbp&*}1%~|ENadt6{BFUm8lhQ;=lWAV(pN)$sTpJ+b"
        "HlP86Tw*H7rupuRPhjfhK`$Kgcu;|`IHvX8<1gSMQ@*dU)#|6y`uy(5I=O9)QpAxUzG$SGaiBh)md@UlPRsUA9TJX`1}F%qq?T49"
        "q#NXd%a|Q#jL}_q*>6samkoWR1r!Xo44ioAwimNlW;j$q6k{u2#xp=ra{*^)%%#P~hJBAqESNp8tA$vkx6stG7OYforNz-^x!14X"
        "B!fvb9~)gKo8At*sV5`ZcJMct6nrN+3vPL<_34Woj?Gio9Ort)n6M4Vwdk)6mAb-FuEk}7f?Ie!;KBjKgYom8N{FQ!cTBolj90Y)"
        "#+zoS(Q3d4>X|Cj9_eYKkjf0P)*5BXGcOWPVwU*h(p63{DXkL^LYzV499SF(b?&hX!yObqom6(lrMD1jlfsWwfd__`Ni*Q;N4{ZE"
        "z9a3D)^#3nO}aLGEg>|eR*ziYdyfUx4QP(DosGA2@>$Q_y2&zh^8^6h<uFsn$(%D$+tWNN%?0NjhK~LD&#Xuj#&dNccZ%m2kADkK"
        "7uz9El{Vy=3FD*ix2V9y-dC67yz!5zX!%yfrHZTjn2akU)Vid758C8;67kOcem}EX$Jz11yuI9uB6mejhtB;;$4QD<>G6C5d=!2R"
        "pXT8yOZVoK+1x|<j<KHc&sEz)H=R6VbaGyaQ*E5OmwWnKUg>7c`6m|PdEXcrqGy-3a?KnYE8wnVWhc}MoQ-aXG+iIWtnl2R3mXCp"
        "QoRh_tZkg$D}0>=!-=wbA#Ae!nGm9b6gAk`h4(sWC@@PtW9=)9YW>uv!vb&C7I7s%zPtG~U1pCLvrYHFaH?tMQhAEeu|-?lsYfUM"
        "w$g~7kDI-uNslB}!9{p(HRv{jD;q<p-2&H)VD7^nWa5|+{HRy5Np*DP(B^%K&8o=*Z;XV-7+&u!PHVi$8GPHMpB3CR`qF$pOlG%7"
        "$6PI{t?a%q+{U{UjW$$!3fx>@9Yn!Z)xMR0PNko~iMT1X14{M&Lj`l+eJ8VNKelpf#&vp4FpG-JEwHlvl`cN0w{1OqiX%K&PvPYM"
        "<6@`1s-=Q&nbKoKy*!e}@^dth;!4tR$7U6s?7R2>7Jz@zSVM0{-ZAscH3c&fU7&Jr$pt}1&T?2|;5FGK9IF-v-0jRnCj`5r(XPz!"
        "OaAZTMfYOZ;x!KO0c*d-nKe#e5J~-gmW%wN{B}WdBilFRqJVpaUz}_mO%s&>m^3k4$S028d*qep5}HQ4>BPlRfQx8RNE-0-v;P55"
        "O9KQH000080000X0BnGk3yA>$0Eq$s044wc0CRF<FJftCZ)0I>XJu_|VJ~!YVQFq@Zf7rWa&Km7Y-w(1E^v8ml0j?SKn#WN2LA`)"
        "Qvw^i(3=m19FiVNm$FTB4a!*K9Wpa=HF7uGUti6po01Zrgl)a|^hsUU-F^5tLLw}y&;)yw>3~A82>TGJhb_bX?)ocG(U4FHLID-B"
        "Y62{P|9XYW8ny4bu5H@J3nzFCo*l-SQo`PG_F;yQ78Zq@=Y=FB%KCjF_ul`-q4yl8bH}Zjn$ebZM>rjCL|Cd*+f3xofgk8q_oA5a"
        "_@B`REQZpvJ}5s?E+d4j8&H&)-n_ent4}Zmr(@fIJvP?uTKz{jP?+~x23`zF_)RjP)9?#5a4tik=&!!q4^FrgE2?9h2BVrF8n*0?"
        "eGOejcXXQUOnqZhkPF3hEK<z^k@Bp2<D3n5j7CRUU?F@eP-sCh#dw5D??sZlr<%rAXde?PN$KJ4!0`Fo4{*4k3mV82ZOfU}mRQzg"
        ">r;kDYhsQx^{+F(6oR49y9>Ce2h?q#+v3SI@kFumqghTs>b(g4Ttht(Z`qC2it+EhLZ18XvTgnVP)h>@6aWAK2mk;8Apq#gwV%KU"
        "001u?001ih0047xV=rQ9W^ZF*Y-eR{ZDB8Ta$#w1X>MmPa%FIDb7^#Ma(Q2KW^^uad8Jw1Z{s!+e_vq#2UhnYwOU^sw0qYDd=GcX"
        "0{75j(G7|MK0{z+$|j<(BPqvjkpI0i{1Pc!&TgxRL?SuEnc<JW8PZ9T{KEQ9a@C0wVaL7`im2|9wfuLgc-IozZb?H`xo6vVeOL4H"
        "L_UAKCsij1t!pCrk(6Cq@!cS_fLtU=^7hT!H<job(o?ms`2*pN*8B-RecuDGx{}sA)--fsbhvSMzkRy<<-b4O{d`}1{&e@B&odHA"
        "&{9b!bZ^XNTkwc-ad)ImU$aZ1TdL@ytE!r}tdNy*8q&9K2t2(21%ef9%N1{Tg<xeTwlX)pvah+C>r?chW<>=(O!R`c3Oce`6T}#L"
        "Yk^>NTgYw@B`ZYtBxgmVHL}8W2f_y?c*WVaup^kbqyr+yTd?v{6e@5pXuc|L5n&}A5LjW_i>BMMIwxCJ(V<p_6DxYrRlH^rvGDc{"
        "bUVy}*FGjTwsuh<8ac$Cl{sN;33>(UwJ+cL#-ip8SF7)DethryZQDPWKi<M>d@%ZWg(??2rqV=oHA^z`&59)2#Kex#^HMGu`IXiK"
        "yAz@lX~GWhp#$$$zd*FKUEm<DQIZ)g2vl}dWQ7y16Z@HLOs}(>ZzYUS13)&lZ42{1wGug@TL1@yfYaJ6uUy@Wfn^zbgB?~v5LScH"
        "Y35xGI|Jun9hx33OUE{RfE3gqTGF-kiBK}M8d7A-tBO=Z4QmJJ*b%j7gtt<Oq0~4Wq-%9{Sko8(Ym<0}?@TZPX%`TDtl{XzC2Rkj"
        "d0+&X0W({pjN*n#Np}o`zkW7I_&}0DGLhuOELN+a$xQyT@f1-3^Q|1j7PRl{Q&H|IfPew-P{o3nx=C`6^VLTR&Y30Mv6dO~EX*u="
        "RT0vfP1K&ZkKm6HF-mQKz}fn%jerOQo>D0}&=&X@xvN)4Q33m-=#l=D&C4bql}-E@J1#pk9d*&QJo3$iINBG^PrEP7h>P1o>&t>D"
        "UDJd83jsLceVV<lj#pme`U9z1n~tVt8)T93`zn#xVK&MS-Oz5SI62xd&0w>iY%&%3u<-@~`Gf(3<UD?@Z?sXj*oxz<V)!y|>vu~%"
        "V0rWYs<{C(N8zz#pN!!_LjzvR$iHHG#<_8f^DHCp$PxxceEp@i7#ljsdsW>%IV*%=h|4uW>bZxukZAcf+2kP+=Rx#x8>MW_Rjxh="
        "jj+V;DhQ}^e7OKA%dV))$VEw^p0cgq=5Znj5}4^Y&tgs=PHLNPZ*Fe<_`}%ftYB=rTK*#+^PekRAl=uB3EUhrbx%Mc)AA%mM`6pD"
        "rd5KHJuM$AWD`N3tR2f6e4^cLF$*PRDr^$gvlZ&pV=i6+%Lnn*6jU6Ri7q?^(7JOVyNZ+hJ(olbE$|Tt{D0sA_)knLUWfu3)9@o?"
        "k(%u&lr&cT!U5`U<fCBhcLqrZ(p$}U90&|j$sP+IT8d6eUl8lEN0J8y#xk-K-Oz(#T{vQ37f$29sbpAYNwTL02Am6Z7W#*YOVWms"
        "1DAFhQ$qChARQhoE~V9`Aq9*0{DLqwIH(A)I}Awy%}fBmSQss=fL!}Rn+$l@fq>|0b-`di#rj-k1Jlmbp5x59F8<TW9&nr*Y}wga"
        "fXAF`4>oX?PF53o`7Yiw@cqN`)~0)Nl*!#oF~WZhB!JP=DFac~Jq0)mVxh!o5*Zv`!byok-G?>D*#TxUK4gR0ig)^81$?>3Pd(Fb"
        "QQ(KUN>JC0XRW4q{26fmrRr?7vzhTRvqfJs5;5h)i#cy{jEW1Yv$)AZ0Lv!UZZRnY1K6=-P;8E50&WwAFO#(dPLxpluHf(|Qa#}E"
        "G!AKNBY>M`(xZ9M^ycIy1K2Jjt{x=<{>M|!0cXE-ERCq!^1V(4_$QANNbsHqDa5};?ClR;@KQ7s>Tf316&BFP4=;3$2>jU&UP8@*"
        "Cs!x&@~^SY<WLkWygoW<`0GuOMV^;Xr73B7Rxg69Ff@aV?4j2KsURJfDP&(}N)Z`|IZ;bITEhfv=hPGGa!9_sslJ+UG;C%`2cMw0"
        "x%9Zjm>=OV@v4rmAdf)r9D$slm0#r!e6m+5<%?MZpI4a!9|O36G)%)RSK#9@=>XvXa$bD-q1WrEJ2=Fh3}5^m`G!^sD-aa61tiur"
        "u2u^6?enNL0c-<*;h{4?pe=0*JiKFp7lb}BoMw*~AO^hOwY-FbN1#BLSH-@ox80y3aa&Ou`w9~`P-EFP9O;PIlyhei6P~^z2?quk"
        "zZR3z#aY!F9;Fj=f|*WgUZQXSX+wwR?w~o^6VN}mN0Mk8z`(z<Es~L1O*sJio}DSi2v6U+pW007J6=MCE?`>I<T}dU%8~VP5yVwT"
        "^@vgRD25twGugZ__u-2~myu{>n?3UM0*;w2sz8PDG|%T-_t9LVJZ{draQa7`9U`FQZmRJOppK`2uGLK~?d85xh1}EIAKqh=hJP|4"
        "j+gWDG_GdH{&OT^Fv2TrgxuM^_9SJj&s|)nQN(j{Ttr9gqbmrC@UVSc*y|8|E#Oy)E7r)FIO@J=DdTbBy`pvYCFZGh61uz5jC|&>"
        "8ysISgviSyFy&tTcyoYP>)|y@<>F{A%EPkGpXKcvkXt{v9q4yfe1*`^V<8pmC7!t5wL;&OduXtqv@C~asP*OC>igXJB8LXN6<xEk"
        "b-zmo6>WHZiVHALBzXTC-0K2^Xg@rKz}XwsO>3(}o|=X!!OLWn(cGgrD;!ANBu~2Z)shJvCF8Qp{tmI4|EVF(*2b5OE95TyR*~9Y"
        "EKHlP`I;UA_P`+<ohCVrigFJCECCGqI<<v?2R2#cX<lQ;9uj6_ZdwEPTR8KF<BY+Wvh+*D<y6thCY?cZw7-2+TuwDF@wTwgp8O6J"
        "-XsTTogy|yEk3#uH}DNlh*4C7_7TMXExb53a5n$KYv&Nx;m6d>(fd{8`*a@!@8iAnA5cpJ1QY-O00;m803iU|0ZiW{0ssIX1ONae"
        "0001Ua$_%IX=ZO@VQgn*ZEayMbaG*7ZfS03FLZKYX>Mh5E^v8;Q%!5zKoGqf{2vCL5>RE+rWZj<OJbKCVnSl*QH*zI?WOyn?ng*M"
        "e|%?U*)GMQI!LSCw=?tJys4^cJ^$E()PY2Tr{x-C(9aa0Hz;W5jik`P6A?%+;%xo+?QV6q`rcUCR8=*dOeZRM3lb9wgE3MBZF?Wm"
        "X={Y^n07{Q4#yAtd%V8U$_q4GvKAZRozmL_Zt>Dn(3V_!>Pei#M}bOlp@$nw!juE8e0n65uZs|TD4C{{ug97$ILsTl>vhO<Ih_o9"
        "h5gNtyPZyeA43?NcH1ssq860#nBX_8yd$_CK6{+iBKU5DN$k3)-x#XhS>QIH)XbzqA+pT|llHEQiTe2N9g8pjLG;`@9XV19F@>`T"
        "Qi+2U8r#=*QHfCOD0G2-WwwKCVdK4F$P!JYy@HYya2)Nry<@tl(#8II<o2}xw5uLm&{=d^QB3X7^&0jaZcl>O;B)G8DpA|f8kg|l"
        "%TV3e4Yu1dm&Uft*eh+Q03w-U+@U7i@LrDrEhP%E9N#R;n`w!ky=VQJLZU>woqw9$&aVKSg!}o`?0)_kY>o*w1fkf57s>|$-zROk"
        "sbN;A0DC~PB^p@ocqB++Hfa=@s|H79b_{06IV0df)CN}B7}&kCI)7C-N)47UGBY_cD+!LI_Y3A1>8!O!`3E%5a#ch1qr@!b*5KG<"
        "&z!(ZGBrFbmh1XNX<&a6hs%lCQZuIHx%dN6O9KQH000080000X03a=r2c-@G0NW}600#g708B|kMNU&i-CA37+r|}s&rJTq4rkht"
        "W+2*ele9^mWJ`|OI+TZ$>&fF1Sdd!-SY#Iv#s2!9%icgzvO9gN$CFqP_Uzen`ObIFLWnQ?x$b4TQ{ufYRp->V-#qi`Kb3XbbmG<g"
        "Wg-49yHVPMc=ht-_fN0d4MYF)ix>O-elBUFxv|?94R7XNeESV8y}w=kwiegR8}a6Dd9%1*+%4DQ?cGXzT;CRAb^Bp;ck}TLJ}&63"
        "o5lKmwRrszpLh}9&BcwXb*Bf=(9Qi*_%rh-n+dm*O(R+*J2AkC2W49)WLJsObQP}?wXtG!st{K7)>LDOkBcA+U#_$ptlo_H3n87T"
        "a934{%|WbH$x^<94XhcrJMoh+HT<CAJyVXY>V{L*j6Kn<H2uNq?QRfe-zh6VD_E+BgOKB}Ggkkbb_{~fEg5!l5U}soN?6%#=_PNA"
        "i4JNj8*xk0PxKf&JS81L30abypa|Fk-VJgLyy2D9%JJ@C*TI^m5YnpfLqm!ecp`i{b`>luP1~9-i1KfWy&iTfh_{@Jx5knt{b+jw"
        "@so@jT_1fl^I~Rn8YeFFC9gGmWeael1#h7my5oNrVlbkV;C*~Ah~eLn6|87wC$|c{h&#Kn+<DasvEM1O0E7W;EJ^spI(v<Q1!68V"
        "=t5p{JKdwGTGwFGUfB|5U;Oa$PnWcx0TZ$Hqoi?g1K1ut0nW0@1(6_gqdG7|sllNW@tLN%>wh=nOkBWf{Fj|w=H7?DXqX3GjVR5E"
        "JWhfH^#~ekhq{37t#*#WpF)mPg1mDk>ejGf2{{S*)*c64Z&j_V1p)NinhgILH)&0!;e3*^EP7gZWiygNAS*>@2GQtNqXOViXX;^("
        "k>+Sea83nI2+2s2f*gLM2#`?g?P%!*a8IKqm~yxI7evLW`m#IlGw{pUP`cFCwD8MvCp*wQ<_knf=kQe-f{UIs{>NHK!8Rg!#q^Lt"
        "$kACLje3m<W=K7M)-41DoC6+DPCWrNIPL?3yF)>o%&n@l6vLsPp8b=tpHF~d4-ZLef+vi-loGlNM@b362J~lZWd-Q+Aax@*O~8Z<"
        "t_s9ijK)&>u#yQY0_uS+;Oz)|Y%h3QlX2xRVD%w01$~2z3pj&%lx+`t!%Dz?h<aX$uU_}Pg1sLhyBf2<%uR8ltbPD%J}7}!akJxC"
        "z-`VO!doDSVnYPwH`1XOJ4((9H^F3pKw=2Py~u5t8vC6tcR50U*9NE%kgQfcX!0q>G}y=IjZk1`W5Z7%!}~kWt{@HTpmGp2<OB)Z"
        "nTB!%*6FS8V4G9lpCWaHuX+Lrg*Y~$w;)EgcOl956~(Gn>XfjmmzF{XZ9^w%l~v6FlA`-e#@j&HV8C>;RhPl}8qmqsvZMl1<V7@E"
        "_CzhTkTP}d{Wpm7o(Y`w{Bc6W9L~3lX7mXds%=ySrA=Hz@mzVv5=9xd5G{ijo;pC0WypXf!NA^4M07W908Tuf1bRYoM#__dJ}+n|"
        "g7_2i5xq_I@Yzz56B>kT+8;xIqrjjwSmPOz^L30s%;KqLK^Sv_2-mPmHIR<h07w_;ij8b2p!OCQc0^Ie&bvThLQE{LlCjWMgG-4+"
        "PH@GuWh%j0z6<<KS^-Mx23I#gKtWtq&mv=T2REqJO`si$@~E)5lvG0fD;xz_7n##UQkGl3$YEs?44KK%cEE|t(Gd@zEnC7+&pAI4"
        "24+=DJqANej~fCLPUCuAj%IX_%B{4Y5e#i2QIVS|r?(x|1c*uW6j}FdNFsX8mSA)#^1PkT&L#8_BacT9>G)-4l$#MTvpw2Z>_DH5"
        "g1`cLsR-Xe?|h>yBc1wVgg|O=<I;fDxtt?E&T=t8)2q4o6$u#b{3aS9a4@kRx%B$5J=+E4Ntc7Qf*L9^D+$CD(3=P!F$W-$a8e+H"
        "y&AwgF%kg|n`*C-k94NHq7ZO!Z2a*GxRKpr&t(p>84g#qRq&Gr3ix13#P`!`?>k7`GIVON1TwD2m_7l&4CecB1B=0y5VU<GA?o83"
        "P@C(pqbHtv<UQY{HHZjJ^!jAaGu58JCiv+unV&vL#Nz)CCtbig)eo3((5Vc8NP&uuJE=?2^KmmL0>KAqJNcl94Mvry7dLf{gb=EO"
        "Y5<k_zW_4E4(y+Z44zhd`bfwV4~J%8SA~6L-#6H-nhu;#rbayT%9V}OV03;rKOdNqq~s=!Xxc$GJ14ECWUVa#QfP#fj&(6df(v&E"
        "eYEM6ua<y`Kw4rOK<kbd#4~Y+>}xTcfXU!gJ-^2u{SoTmTr6q~gVaJfKrf8%=$b)qSqZt7_-g__-}_&rT9%l&HO^g;)$k;x8Ifi2"
        "&kz7oG;;4oJ>Y2@wdLvnRtxH9oZ$#6p94KA7OcASy?Q37OfEWvlLfD|#A!he^R;P6Mp_kGS)VnbHA(sA%VA(%Ts1IX(T6c+qzkmJ"
        "f~O%0qlrNrwwP6bhVSQMrSd`3oOW;JA)(zdFab8|z<;I?^$Z&%&mg~q-9`XD3Mgbf@PCsk_O$Qj8vg{k6{&wBt7JjeD#h-uO#_`2"
        "6Mz6`Kc_l;Ausv3BShU6bw(ArM*#<GI41&G&XZGPiGNN`E2(lEHy^)nnT^|SvTb<|l&ClMUpNHd8Ii?sf|f&f7&F{?x?BSUz8K*s"
        "6x;tT*+j9R<IT%#bE^hOxgrqlYz{yj1t@iV!hG+zX9|u2b1qfS!bf|7a9Js2SVc}bC}M|{fc}6y$~dFZ3D)AD#PS$cgAh`$GDLu("
        "1mQ{1sGNg#$jTZY=Cqn0o36S<G>fs|+eY;IY<YLTcyl|0gnS&x?wE;w8)WJER(ZZ<z;GsYPV7r=%VmYuRe~>NMJ;NIKy}u*h*c6N"
        "$+;Ylc7zvv7CJ@oHP+4LoHz7Y>r&W(kVZ*|jo9Q$z^_RO3=9gk{W&NqgO<tQ$x74sa?fh<OAhptaGPZMbfP76ozMqs>NZu#(+!L*"
        "&X`&TUYO0zd`EG{QuT4#5X*q}kzEE7ZFPm`JVXz5IP-&!8`+1F(8AvDxPe1ZJ7eq2?Zhy+7mt%e=sZ&Ii#+B?A)jf4Vu#5cKrAV;"
        "s_@?y8>Kunb1^}CZ>6tEih^C>z#EfeQ7eX1lB!Z&HMW7|P9i5D4L6U$sV6W;Rt~ceFvZ!#qS+TTTg(;gcoK+goTu+d&KfH9BE)j("
        "Qh-^)(IhSR2uksX$<@XwpGJ~9VZ~=ie>N^Wb4|r^vg;{RpV6nta;T<W^YBCmlMno2UXeiDJD(6H6;7_mCS^MEbR1O^_r`#lPIzL2"
        "bJRT?`Q-=dlKh<m_qvHQy1Dq+L6vpnJM{>Al^Pppl9^q{q)Ex)m~>{-v1}~%<R~m_4{*=psSp#xO+MZJuev(V$VkyFa##ja-74OF"
        "vVf%-a8<l6qRO)|+|OYeZ>crJ`a;UN5z2&9mEx->%!%AHek&#*JbDFF#$IX*{U9a5!DkCKbLvr*IrI}AN26G^m6orvj$4*^<?tgQ"
        "WMC-{(K6%Sl_5YKn5JiUP-uRCk;fb--k8W1XK|7D;2=%e2fR7(KZ065p84evz(MI4GO5kA>W@+T`zkE>4qSsJpIi&fz_fT#j{1PD"
        "fti$WNbiu;tl$**WcC|!E4a_si8Hk!nCq`|aiblzU3hO?i%-B-!8nJQ5mDRCfjeO8-myK)fPvgeZBLr470GcP4P4Up0@cOQ!%2^u"
        "zm9{^iI*>NQU=9fcD)vh^-R3JUN6?s#Ge-T@9sX{i%-|9)%Egzal00GtNa^*ySL(c`MdbXVtE6cNb?=^BMx3&b{I`qQ)MIAlutB+"
        "l>xgB&?b|)sFkv3kQXd^zqo&YTY&MGSBvG_)nfVU?Qgfs`$GJ7yL$5u^t^t(c)z&+onq<j;(mF%=C2v9gOCr`D{$P$_tz`&;p6JV"
        "-TIcRG2c@*c!L2)=wVw;w;OZ|#7*T<+<*hE>8(aSPlv7{&hQ<I#013IES9I3&H-`46Gs@=jv(KeQpZl4(cWM4(x^AT=RNI)LlpjP"
        "4o`w%aoKw<H@cxq`~{X~0VJs#NEMa>j~bezf|}5+=VP02cLtF(<fDg9Z5s_NtyGsqymT)nlj1Z-`eN)~FqgoagGO(N`H^~CoNJ^D"
        "lCbH3pF=phr9Yo5jH#2lf&-rDxkl60ey~6;maW`QZwhdE_&Ondr{H>pcl`MUIAj;FV7{_In!^)gyae=8190OQ4>ZStxaFIBtn#Ur"
        "<F)m1qf90pBkGL&M0eh$8Q4zd4j0c}%m=;j<c;Cb-5OKvbu*breukQD`d;Gf5n02Cy4O-Sqvb*+n|kaL0iptV?%Nc+Y`{3qZNghC"
        "2LXfuiG==mwikrNfxWCAG+ocu{(BP025*S)sfCvxvf-z>xGu3+peX_#<L1|?kY@?<X@^{Yn!P8Nl+RxA1;SeHjN$nzjZmjo{xr`8"
        "f>$d-7=S<0QFf)`gY-OL^@w<&2vlu{pSxr;Ha2b()E8#c_^}pon->T~NR#<q2abff<J&SFVKp`-@5~<iC+<R{(a5Tq<a9*(&ZcYf"
        "i<d~z{S6FFi2Y+k*MzO4F|na^VUq!U8ark2;3v-5YU!F$8}pyjpX^g7`&6oi9*36zkFL%P=cR24ngSJ%mdvtyv^L$)`N1?GH8l0u"
        "=<<BFIGsvw4xWu<XF8x2l3^qH-)CW&@yn=9h~?Ym4OZQAU+noe(w)<*m#=<=T(G10Jjjjsvv{p*Q_AL7)wVL%)bUCm1Xae7r(Y))"
        "`6r0t0$+%q_g(%%G6MTVpE4-<;cDP(y7SEk1$vwhqyi>V!)G=$HIb9yO#nqXqz>^*-#NYqnzmfuUV++v(btf%rz(s8q>1EY<CW77"
        "XK=8bzwjj2`X7l_U|4LuwNeu$!i6!FEoejGblr9YfCRjURPT;I9}E3*qGs;&Xt33P08mQ-0u%rg000080000X0O_tvAJzc?0Ei6$"
        "02lxO00000000000Du7i0001EZ*FF3XLB!QcyMKMX>Db0baO6hb8l`?O928D02BZK00;m803iT6)^XEd1ONby3jhEe0000000000"
        "0001h0Sp2F0Ap`%W@%@0FLGsYZ*ysMZ*qBGb98lNc`j;mZ*EXa0Rj{N6aWAK2mk;8Apl(|JEmp<008X*000^Q0000000000004ji"
        "rv?B3V{dL|X=igUbaG*7ZZ2hbVQp}1WiD!SZ*EXa0Rj{N6aWAK2mk;8Apqx#7AAWE004;s001cf0000000000004jiMF;=@Wq4t2"
        "aBO9BFLGsYZ*ysMZ*qBGb98lNc`s#lVQgP=Z+CMpYIARHY*0%90u%rg000080000X0OeyG4Z;8b02=`S03!eZ00000000000Du7p"
        "3IG6QcwudDY-MvVa%FIDb7^#Ma(Q2KbaiBTFH%K8L`_95ZDdeO0Rj{N6aWAK2mk;8Apm<V*EZk)004^t003zK0000000000004ji"
        "3JU-LWq4t2aBO9BFLGsYZ*ysMZ*qBGb98lNc`tKrVQ_P3Z*+4nF=A$9VP-ZqI5ajnGB9H`W@b2LHZx^2IWjXfH85f^I50IeG&wkB"
        "GB{;6Vl!nnH#Rh3GGR0{H)An3Ib$_6E^2dcZcs}B0u%rg000080000X0KArWn%)2a0E+<t0BHaK00000000000Du8;3;+OScwudD"
        "Y-MvVa%FIDb7^#Ma(Q2KbaiBTFLQ2TaC2yHbaO8;WnyG8WI1DGF)}k_F*7qYWiw(qHZ?UfWo9%rFfn2@WH2){W@2SxIAJ(BGi7Em"
        "HeoPkIbmjHH#IOfF=I6@YIARHP)h*<6aW+e000O8001EX=dStY-~a#siva)tX#fBK0000000000fC1JG003opVQp}1WpgibWpHnE"
        "X>@OLd0%sMb!2%jb8ca9b7*gLb1ySwG%;c_WH>P}Wo2eHGh<^oVPQ2gWjSOsWHdE5G-fqoGcYt|WHDhhFl941Gcq?ZW@Te#Ic709"
        "H#jviIW{h8b8l`?O928D02BZK00;m803iS_E7hIh0001s0RRAL00000000000001h0X+`@0A+Y#ZE$R5b1!mbaBp*IbZ>HbUvqSI"
        "WO*-hZeeh9Xm50LFElh^WMnfjGBsslG%+$ZIA$?pG%{i~Fgav4Ffn8>Wi&N8W;r-9GB#p1WMN`5I50FYI5;t4VPau1Ic736GA?R!"
        "Z*EXa0Rj{N6aWAK2mk;8ApjPL*g@X_004^t003zK0000000000004jir4RrBWq4t2aBO9BFLGsYZ*ysMZ*qBGb98lNc`tKrVQ_P3"
        "Z*+4nH8?gfV`5=2H8DA6HDfg~G%+(ZW@ctJWH4khG-fqoIAbw4G&49iVK_N9Ib<+pVPiBoW@0&IGh{h2WH>ZAE^2dcZcs}B0u%rg"
        "000080000X0L%vnVc-A&0Ez(s0BHaK00000000000Du7t5&!^YcwudDY-MvVa%FIDb7^#Ma(Q2KbaiBTFLQ2TaC2yHbaO8?IAUaF"
        "G+|+8Wic{2Fkv!eVq{`4V`MZqF)%hVW;ruuGB{#mH#RvlHD+UGF*7nYW;Zl3F*jl{Gd5#oV>mf3YIARHP)h*<6aW+e000O8001EX"
        "Gdt}O-~a#siva)tX#fBK0000000000fB|w7003opVQp}1WpgibWpHnEX>@OLd0%sMb!2%jb8ca9b7*gLb1ybFF)%e{IAvusV>vQ1"
        "IW}T5F)}w{WjHW0HZe0dWie(sGh;G0WiT)}WHT~kH90sqWnpGyH)3TpV=y-`WH&Bqb8l`?O928D02BZK00;m803iT%jzm!40001s"
        "0RRAL00000000000001h0ofD)0A+Y#ZE$R5b1!mbaBp*IbZ>HbUvqSIWO*-hZeeh9Xm50LFJd`1G-P2pF=S(5HDfh1I59afWnnZo"
        "H#B5oV_{}EG%`72H(@b0G&M0WVmC24Wnp75WHB@~HexY1Wn^JuW-e-TZ*EXa0Rj{N6aWAK2mk;8Api}^5s}~k004^t003zK00000"
        "00000004jiKo$T1Wq4t2aBO9BFLGsYZ*ysMZ*qBGb98lNc`tKrVQ_P3Z*+4nWn^MGIWae4H8o*mWMeckFfn3eIWc25GBY?ZIXE^o"
        "IA&#HGG<{oH8?O~Wn*JCGc;y5WMW}7F*ag1I5lKsE^2dcZcs}B0u%rg000080000X09Mc2<wF7h02mJd04e|g00000000000Du9f"
        "7XScdcwudDY-MvVa%FIDb7^#Ma(Q2KbaiBTFLZKYX>MO~Z+CMpYIARHY*0%90u%rg000080000X00g{E!)5>g0BisN02}}S00000"
        "000000Du898UO%ua$_%IX=ZO@VQgn*ZEayMUtei%X>?y-E^v8JO928D02BZK00;m803iVLCc-_F1poky5&!@k00000000000001h"
        "0n!=(0CRF<FJftCZ)0I>XJu_|VJ}}_ZDDC{Utcb8c~DCM0u%rg000080000X09jot%})RT09XJ303rYY00000000000Du9XAOHY!"
        "a$_%IX=ZO@VQgn*ZEayMVP|D-bT40DX>MtBUtcb8c~DCM0u%rg000080000X0Alb-By#}(0GR>+03HAU00000000000Du85Apih#"
        "a$_%IX=ZO@VQgn*ZEayMVP|D-bT4dgZ*VSfc~DCM0u%rg000080000X00oszHAnye07?J=03!eZ00000000000DuAAA^-q$a$_%I"
        "X=ZO@VQgn*ZEayMVRLh3baO9XUukY>bYEXCaCuNm0Rj{N6aWAK2mk;8Apn-6WlZA&003bM001Wd0000000000004jiW+MOqb8=%Z"
        "Vrgb?V_|G(Wo>O?FJW_YWpr~dXJu_|VKiT5GGZ=jb8l`?O928D02BZK00;m803iUlta6Cs0ssIf3jhEo00000000000001h0gfgB"
        "0CRF<FJftCZ)0I>XJu_|VJ~5Gb7gdMFK1<KZDBNDWi(<gYIARHP)h*<6aW+e000O8001EXgY6oG9RmOWFA4wvF8}}l0000000000"
        "fC08C0047xV=rQ9W^ZF*Y-eR{ZDB8Ab8}^Mb1!snYh`X}dS!B7ZDDR{W@U49E^2dcZcs}B0u%rg000080000X0Looysg3~v00aX7"
        "03iSX00000000000Du7?EdT&>a$_%IX=ZO@VQgn*ZEayMV`yb#YjAIAZggpGXD)DgP)h*<6aW+e000O8001EXhr*_*d<6gi4GsVR"
        "8UO$Q0000000000fC23;0047xV=rQ9W^ZF*Y-eR{ZDB8CZ*FF3XD)DgP)h*<6aW+e000O8001EXBIo{RF9QGoa|r+d9RL6T00000"
        "00000fB~U20047xV=rQ9W^ZF*Y-eR{ZDB8CZ*FvQVPkZ2E^v8JO928D02BZK00;m803iTX1w2xN000280000X00000000000001h"
        "0Sh<)0CRF<FJftCZ)0I>XJu_|VJ~E1bYU-FUukY>bYEXCaCuNm0Rj{N6aWAK2mk;8ApmE;9qPLR005~4001EX0000000000004ji"
        "%{Tx6b8=%ZVrgb?V_|G(Wo>O?FJxhKVJ~51VQ_S1a&s<lc~DCM0u%rg000080000X0H)DR!O#H!0BHjN02}}S00000000000Du9<"
        "I{*N4a$_%IX=ZO@VQgn*ZEayMWMOn+FJW_ZE^v8JO928D02BZK00;m803iV5)`me10{{TW2><{g00000000000001h0nt4G0CRF<"
        "FJftCZ)0I>XJu_|VJ~E1bYU-JZ)|K~bZ>GlaCuNm0Rj{N6aWAK2mk;8Apkx|#|_8^000pZ001BW0000000000004ji9YFv9b8=%Z"
        "Vrgb?V_|G(Wo>O?FJxhKVJ~BEZggdMbS`jtP)h*<6aW+e000O8001EXx4zh?Ko|f3piuw-ApigX0000000000fB_>&0047xV=rQ9"
        "W^ZF*Y-eR{ZDB8DVRT_HV|8+2bZKvHE^v8JO928D02BZK00;m803iT=_*tYj2mk=T8~^|$00000000000001h0is|40CRF<FJftC"
        "Z)0I>XJu_|VJ~E1bYU-KX<=t>Z*z2MV{<NWc~DCM0u%rg000080000X0Fb4puwE1Z0Lec903rYY00000000000Du7+X#fCoa$_%I"
        "X=ZO@VQgn*ZEayMWMOn+FJ^UYY+rD1Z)`4bc~DCM0u%rg000080000X04a^&e47dY0PG_G03-ka00000000000Du9weE<M+a$_%I"
        "X=ZO@VQgn*ZEayMWMOn+FK=dMY-w&~Uvp-3E^v8JO928D02BZK00;m803iUuN$^2S2LJ%}6951p00000000000001h0hNdV0CRF<"
        "FJftCZ)0I>XJu_|VJ~E1bYU-WVRU0?WpgfYc~DCM0u%rg000080000X06Uc0PhJfG0PQLO03!eZ00000000000Du7=kN^O4a$_%I"
        "X=ZO@VQgn*ZEayMWMOn+FLGsYZ*ysMZ*qAqaCuNm0Rj{N6aWAK2mk;8Aprjuz0CIr005>O001EX0000000000004jiy`2C6b8=%Z"
        "Vrgb?V_|G(Wo>O?FJxhKVJ~xcWpiz5bZ9Pcc~DCM0u%rg000080000X0MBF+v$Y2R0M!)$03`qb00000000000DuAWrT_qQa$_%I"
        "X=ZO@VQgn*ZEayMWMOn+FLZBfWo~JDVRUJ4ZZ2?nP)h*<6aW+e000O8001EXr3N>f=>Px#+W`OoCjbBd0000000000fC1~S0047x"
        "V=rQ9W^ZF*Y-eR{ZDB8Ec42IFVRUJ4ZZBV7X>MtBUtcb8c~DCM0u%rg000080000X09?f29bpFm0Q46C04M+e00000000000Du7*"
        "umAvaa$_%IX=ZO@VQgn*ZEayMWp-g~bzyXAZ*DJTc42H^V{c?-E^v8JO928D02BZK00;m803iVU1ZD}Y0RRB{0ssIe0000000000"
        "0001h0l>Ea0CRF<FJftCZ)0I>XJu_|VJ~HNVQh6_bZKvHFKA(MZe??GE^v8JO928D02BZK00;m803iV8>g^7s3jhFcB>(^{00000"
        "000000001h0kXLO0CRF<FJftCZ)0I>XJu_|VJ~HNVQh6_bZKvHFK%IUX?A5_aCvlSZ*DGdc~DCM0u%rg000080000X0JanEsMi4i"
        "02TxQ04e|g00000000000Du9Y#Q*?va$_%IX=ZO@VQgn*ZEayMWp-g~bzyXAZ*DJea%E&`V{~b6ZgVbhc~DCM0u%rg000080000X"
        "02JkHTtEN-07C!(044wc00000000000Du9%#{d9xa$_%IX=ZO@VQgn*ZEayMX>Mj^a%FB~WiMY}X>MtBUtcb8c~DCM0u%rg00008"
        "0000X04kA3%HRh80OT7004M+e00000000000Du8Q$N&Iya$_%IX=ZO@VQgn*ZEayMX>Mj^a%FB~WiMxCZe?;|bZKvHE^v8JO928D"
        "02BZK00;m803iT%;)EM10RRBx0ssIb00000000000001h0cp?x0CRF<FJftCZ)0I>XJu_|VJ~fOWMyn~FJE72ZfSI1UoLQYP)h*<"
        "6aW+e000O8001EXIFgjua|ZwbJ{JH0ApigX0000000000fC1Cd0047xV=rQ9W^ZF*Y-eR{ZDB8MZ)9a`b1!0YX=G<*E^v8JO928D"
        "02BZK00;m803iSgX`>3i0RR9)0{{RZ00000000000001h0fpHB0CRF<FJftCZ)0I>XJu_|VJ~fOWMyn~FJob2Xk{*Nc~DCM0u%rg"
        "000080000X0LVsj7F7fQ0Hq5604@Lk00000000000Du94+W-J_a$_%IX=ZO@VQgn*ZEayMZEs{{Y;!MTXk}w-aBpdDbYFCGVQzC~"
        "WpXZXc~DCM0u%rg000080000X0Qo1z8*l>v0D}nt051Rl00000000000Du7_-~a$~a$_%IX=ZO@VQgn*ZEayMZEs{{Y;!MTZ)9a("
        "XL4b1XkT!0Z)#;@bZ>GlaCuNm0Rj{N6aWAK2mk;8Apq*<Bm2k#001Qi001Qb0000000000004ji;Nt)Qb8=%ZVrgb?V_|G(Wo>O?"
        "FKusRWo&aVWo=?*WMpY>XLBxac~DCM0u%rg000080000X0JH>t$`S(r00IaA04V?f00000000000DuAO<^TY5a$_%IX=ZO@VQgn*"
        "ZEayMZEs{{Y;!MZZfSIBVQgu7VRUJ4ZZ2?nP)h*<6aW+e000O8001EX#-x%R#|8iZixvO?D*ylh0000000000fB{760047xV=rQ9"
        "W^ZF*Y-eR{ZDB8MZ)9a`b1!XWa%W{^Utx4~Wo~q7Z*DGdc~DCM0u%rg000080000X03MT~bg=>e0D}eq03ZMW00000000000Du8h"
        "@c;mGa$_%IX=ZO@VQgn*ZEayMZEs{{Y;!MdZ)9a`E^v8JO928D02BZK00;m803iS)2#4SR4*&q9GXMY~00000000000001h0Ymfv"
        "0CRF<FJftCZ)0I>XJu_|VJ~fOWMyn~FK%IUX?A5UaCuNm0Rj{N6aWAK2mk;8Apomc;kJJV002!F001Qb0000000000004jif&&2n"
        "b8=%ZVrgb?V_|G(Wo>O?FKusRWo&aVaB^jIa$#w1Wn?aJc~DCM0u%rg000080000X02N1gdYc6R03;9q03-ka00000000000Du8R"
        "3jqLga$_%IX=ZO@VQgn*ZEayMZEs{{Y;!MiWoBh^Wo~0-E^v8JO928D02BZK00;m803iT_7_N^!1^@uN5C8xo00000000000001h"
        "0U;3q0CRF<FJftCZ)0I>XJu_|VJ~fOWMyn~FLG~nbZKs9E^v8JO928D02BZK00;m803iUz0#RGu6aWB7RR91a00000000000001h"
        "0iG8D0CRF<FJftCZ)0I>XJu_|VJ~uJaBp*IbZ>HbUvqSIWO*)dc~DCM0u%rg000080000X0JZv$Z{7s}0ND@#02}}S0000000000"
        "0Du9$ECB#>a$_%IX=ZO@VQgn*ZEayMb7Ns<W^Zg{E^v8JO928D02BZK00;m803iU|PzfTA0002J0000b00000000000001h0oF1B"
        "0CRF<FJftCZ)0I>XJu_|VJ~!YVQFq@Zf7rFUukY>bYEXCaCuNm0Rj{N6aWAK2mk;8ApmqWMmg*S005v7001Ze0000000000004ji"
        "r!xTnb8=%ZVrgb?V_|G(Wo>O?FLZKYX>Mt5XD?%Qa$$63WM6Y;bS`jtP)h*<6aW+e000O8001EXg{`{f9svLV<pBTyDF6Tf00000"
        "00000fC1S#0RVGyV=rQ9W^ZF*Y-eR{ZDB8Ta$#w1X>MmPWNCABX>4p^bZKvHE^v8JO928D02BZK00;m803iV0GGA+B3IG5eBme*@"
        "00000000000001h0X{nc0CRF<FJftCZ)0I>XJu_|VJ~!YVQFq@Zf7rTWo>VAd0%jHZ(?OGaCuNm0Rj{N6aWAK2mk;8Apja<yqj17"
        "005K%001Wd0000000000004ji=tThlb8=%ZVrgb?V_|G(Wo>O?FLZKYX>Mt5XD@GJYGq?|X?A6EE^v8JO928D02BZK00;m803iS^"
        ";~(v?5C8y|KL7wI00000000000001h0fk2a0CRF<FJftCZ)0I>XJu_|VJ~!YVQFq@Zf7rVW@c<@Ze?F{W^^uac~DCM0u%rg00008"
        "0000X0JJtA<`W120A?Hj03!eZ00000000000Du8|SOEZYa$_%IX=ZO@VQgn*ZEayMbaG*7ZfS03FK};cX=8aVaCuNm0Rj{N6aWAK"
        "2mk;8ApmTEmJ5ji004;s001Tc0000000000004ji&tL%nb8=%ZVrgb?V_|G(Wo>O?FLZKYX>Mt5XD@JaZ)Rz1X>MmOaCuNm0Rj{N"
        "6aWAK2mk;8Apq#gwV%KU001u?001ih0000000000004jioMHh0b8=%ZVrgb?V_|G(Wo>O?FLZKYX>Mt5XD@PPaBp*IbZ>HbUvp-3"
        "E^v8JO928D02BZK00;m803iU|0ZiW{0ssIX1ONae00000000000001h0i|mJ0CRF<FJftCZ)0I>XJu_|VJ~!YVQFq@Zf7raa$#w1"
        "WpXZXc~DCM0u%rg000080000X03a=r2c-@G0NW}600#g700000000000Du7tZUF#HNkc_WQ$<iq1qJ{B002k;NB~tC007l|0RR91"
    )
    bundle_sha = "ea156abe8598b9ad578e818d4dd620d5797571acae4de8092f41f8ac13504102"
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
    from bifocalgemma.training.repository_sft import fit_repository_sft, prepare_repository_training

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
        prepare_repository_training,
        fit_repository_sft,
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
    PREPARED_DIR,
    EXPERIMENT,
    TOKENIZER_CACHE,
    CONTEXT_LIMIT,
    TRAIN_DECODER_LIMIT,
    Path,
    get_experiment,
    prepare_repository_training,
    preparation_result,
    mo,
):
    train_data = dev_data = training_manifest = None
    if RUN_TRAINING:
        train_data, dev_data, training_manifest = prepare_repository_training(
            Path(PREPARED_DIR) / "train.jsonl",
            get_experiment(EXPERIMENT),
            Path(TOKENIZER_CACHE),
            encoder_limit=CONTEXT_LIMIT,
            decoder_limit=TRAIN_DECODER_LIMIT,
        )
    mo.md(
        "## 6 · Matched cohort/tokenization\n"
        + (
            "Train/dev prepared with repo grouping and all-six-model eligibility."
            if training_manifest
            else "Training disabled."
        )
    )
    return (train_data, dev_data, training_manifest)


@app.cell
def _(
    RUN_TRAINING,
    RUN_GENERATE,
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
    fit_repository_sft,
    write_json,
    torch,
    mo,
):
    training_result = {"status": "DISABLED", "reason": model_guard}
    if RUN_TRAINING and model_paths:
        if Path(RUN_DIR).exists() and any(Path(RUN_DIR).iterdir()):
            raise FileExistsError(
                "Choose a fresh RUN_DIR; this notebook does not silently resume/overwrite."
            )
        if training_manifest is None:
            raise ValueError("Prepare matched train/dev data first.")
        _model, _provenance = load_local_native_model(
            get_experiment(EXPERIMENT), model_paths[0], model_paths[1]
        )
        try:
            training_result = fit_repository_sft(
                _model,
                train_data,
                dev_data,
                Path(RUN_DIR),
                training_manifest,
                max_steps=MAX_STEPS,
                epochs=EPOCHS,
                accumulation=ACCUMULATION,
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
