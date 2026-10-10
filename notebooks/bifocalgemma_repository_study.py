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
    CONTEXT_LIMIT = 2048  # bounded pilot; choose a new pool for longer contexts
    TRAIN_DECODER_LIMIT = 2048
    MAX_NEW_TOKENS = 512
    GENERATION_DECODER_LIMIT = 8192
    RAW_LIMIT = 1000
    TASK_LIMIT = 32
    PER_REPO_LIMIT = 8
    MAX_STEPS = 0  # 0 = complete configured epochs
    EPOCHS = 1
    RESUME_FROM = ""  # checkpoint-v2 path, empty starts a new run
    TRAINING_DATA_MODE = "curated_pool"  # curated_pool / repository_tasks
    CURATED_POOL_DIR = "outputs/full_repository_pool_v1"
    FOUNDATION_POOL_DIR = "outputs/full_training_pool_v1"
    PREPARE_FULL_POOL = False
    CORPUS_KIND = "repository"
    REPOSITORY_SOURCE = "swe_rebench_v2"
    REPOSITORY_TARGET_COUNT = 10000
    REPOSITORY_PER_REPO_LIMIT = 50
    REPOSITORY_DISK_GIB = 20
    ACCUMULATION = 4
    HEADROOM_GIB = 16
    return (
        RESUME_FROM,
        TRAINING_DATA_MODE,
        CURATED_POOL_DIR,
        FOUNDATION_POOL_DIR,
        PREPARE_FULL_POOL,
        CORPUS_KIND,
        REPOSITORY_SOURCE,
        REPOSITORY_TARGET_COUNT,
        REPOSITORY_PER_REPO_LIMIT,
        REPOSITORY_DISK_GIB,
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
        "0QNeZm2w#X0902103iSX0CRF<FJftCZ)0I>XJu_|VJ~E1bYU-Jb#h^JX>V>WaCx;n{de0olE3%8{tr0j?n)&xBiXU*IIHHRj@tEY"
        "YG-YC+djw3QY2`xrbw2gV!K)Y?>935NDu^Nr+2G<SRw|%U@+ep&~cnM>nN#%Mao&cPSdS0FN<}>I6O?UD;7knq^goUd&M&Tk(Z3$"
        "@^D@AidFf#3^@z)h<lFXeEG$fUnHv{FKf08s%4rin%_y@eEqG;v*t(1o3C|#&9meWco3KQ3I<J69@a2z#U#XGzRqf1j#<RxV4c=c"
        "64nyB2x^>9LB5Bdhz+$Zk~j~8^op-mffogJ;N@|gCK>muxK=&nV*QfC>Z38Mh#r0n&v;psNmdVzR@~LgJoA5F2We8@sR7+DzhLk="
        "`{nrj&CmYZqc`6lpPZc^9iI8`Pu?9J{xTNrA+E`<?y@?#^>2bQg9$6y##cpsXSDg%8YV99BuJ7~1yISCIMhLT#cOCA4PsmJBCnD<"
        "FB?qpiJwF~s{wta+S$pEr-yU@bpG8Lo3T$~CI=~01mSgDI|u3^H6#%{aI$q#rqAa9^d5k|J9ow!NXb7Y$Uz7=+539&BHZ8Kj$gds"
        "VK9BUv;8vWQMCQ_?u(uMINS**dy{x4?hJsaSG@MqB;;Aeahz3BE0BBn*9r60N?|E!Vnue5Tp`mw*Wm{dj>Fx(nC}2yJCkrP3W68Y"
        "7vc6|I^7Nd_r*Tv(PV#@@BJBsGk#kmCZ!e4RT7qYmB;n-V}LtExVK{5L1bTUzu?<Daj+c+d^_5X7Q1`<;ZD3CPxmL&-NoK?FWBAr"
        "db<0W$libX#ryf`4@c+c^SA!t$=kUPJUl->o9i5+^>G3q>XL75dz0qTqKdY*x56}7!xZ|%9rIyQh0XUcoNP_CQP!ElQlN$ZfIc84"
        "<dY7mT>L&dkR2T7=qk%g&~16S3Tl*c+IQghY7ER@2T8_2a(Nk~6$`S630a5}P<0?(DX5Aoq25U1Xt1bVHhz@u^5(}toTKCK$JVyD"
        "KfF2p+x*nlczE)|5A)-5ThsA7>(lv9?~ab=wx*-ww*y1Hoog-k;!j=P9kif8k9C=`dR?U4Z66b$SJj{huewixstmMf)=qN8GoWSN"
        ">_(D%XVmfNMC=Ucpddm8Jn9aW#4tyFS8$f)G^C_;3ne@7+mIKaVw)+?@yom{b1)7^8E#*ImP0-{BMnMFMng+2xd*yF3ewaqoeww9"
        "{&n%s4>upGOJ_`+H5wV5Rsn7S^yeLBlSB56hudS%-hfMhPPu3uvzsKUm$Q8}u;>X)*BeK)KVfG})aA(7c~+zWnB}+(u87N7PzsXr"
        "EMH$OVK5d{6$kJI&%z~`lWUfPVNZj?lT047lxKpz)rkFt?QHi62L|R36vWycbynX#Z@VA~q)*^g%V2wV&viab9PhV0QPW%`?0_Zg"
        "nOOSe$OGJ9f$nJJE%|MfT=5EKx;U8bUXIyxPvaeA!36oG41nOSj`@}t4Mv^H4wCMmajJ@;N<9%PN4B|bb;kTQ2!WhS9$pKjKqGF+"
        "U1I+egkgYwLS7)F0o&ZM1(*>~42wH<_V;(7PduClhx@jn^6zV2Cg3=MvSa8<-7z_!$9_s`__|f(c?xb;P=b>atqT?g89@hPs8y0B"
        "tMzINX^Cx70=k>CC100GRVU#qR`FE;+AD<JxG59#(n@#&#L3rn1)7ajVP0^rS*Su18PWG)_}cfijjMPXkJ-~PtCuCOmU$Y@Cf@$I"
        "ox5S&*>Rq6bewYFJMN=LCLf`=`+1h$&AtN_%jG^CbSLqe34qI+?*>eNRuqZeL!%d`y#$4QZ!~Cw^xBdP_NOLAueqBpu=U7*j1~Zc"
        "E7jON?$CiH70Dk09T+-K_+ZuVXn}UrW~dK!o{>B2>I07YU|s=Ip4Hy!I!a1c{HSK<<r+4C-vV>^`8ECQE8S*)4BWaalqh;(o@L;2"
        "x-=<lQ=WbVA@zdF2Oe}sAZpI*70A`B%t3j=Guy2AA<IEg2XLqB&@Mp_2c^EkFW{)6wXU3z1u;ctfuAR164yDKzndSPv%`}gkI&tw"
        "Blg|t$q&o{tZ1r{7xOv<w-8Xim|TKpn+}YPpE5{5^!^c~Yfjd~b&fzMt-xM@%(c2RLDk_omL&>}*nPg4qQ)Rk*!dEe2mTFMlvF?w"
        "NidjkEW@z?@VsHthhsq0aE<0Sf^A3M@~qP<f>mLe6t&w&-FbU@@}8Z)`S#tM9eu~<KOdc)pON{gUNLK%+OC0S+TT~~@O1v>d@fsQ"
        "I9J6U?0o+7IeUM41a8tV>~Hg5#_XyLRu%j9-O0Bjqbm*PHfGq;w==BST_Nk@4D9ImeEvN+W}Pl$Z8QRFnnxq{)6w}~PkuaSrzbxh"
        "y?y0$={M4qY7j-P(4wllbSX5(Q(B-d*BH`j3Z_}#%<6Zvkb+V(i!-4C5#Nwbt8!Fsw5$q@>X3T9Y-G%$dO0X_F>hD%Nd_NQdOe{%"
        "f=J@@oSmGqBM6gE=eQv!VvE^7-n{z}!VLHI`1J_1Lc8THK}UwvEYh6hi^%C=c;`BQK^@@y-C<A3wGj&k%hn;p=+Std&fmX#b0`rd"
        "Htr$XZa3odz}R54?`%?Nr;OrWW6pkmS1@i%Dbl*c^XOj-b{LI&?5n=w7TY$Dw#EZ}BW<H~S1kP$?_4OLOV}rah45p+a>zI-Lw$kp"
        "mmZ3nhCwTb5c`EoDT)6E`Fj-Om-!9yrNLH~ZI@Q8(MM#(ggpH8*Ez63gU9T3S11gs>jtmeqP!x&2`wS|0e~IJbk~`&@PO*8@xm3?"
        "n4n7+0lk4LB5wHPT*FpVE(s&`6au@*Z@hw%#~gf`5VI5X5S~@ZDoF$Ia_`{b)SKLYt~!hfC4#l%)TKnD194<y+y37aY6`xZ9Htle"
        "5)zm(L<D3y)css0MA^#Y7@dVCW}1viE4B*+)wn;I*c`8Lxz4V2x4XKKqaCPWpJCt2epiZ1(XHntF2EVf0ML1@)G`fjs!H0A{HA)8"
        "=-|f`pK;e4Y_z8kCFucx3u+(jSqJYS7-z)3KRx;JJ^S{TW?q{{Zb)cp8*q|!zX7(Kb`8ie;J6C@&=l;mnRvEQqD3!nUEq#d$l7Qk"
        "T{CO)pyP>69o0<AK^P*}+*~t`Z`SJ53<9kc?3gcd8?MQdkK%d!D6inNSV-s*VjE%Q1<xonh4yWbwrrpgar6Q5dI?TE;spv+cr5Fz"
        "=d(~&gZv2g{5jhn*_{f@#&)N$nqY&Ts8lL(^a6bvXNmo{7{@g0jSdFjPT%JOX`#wsA^1a#z_vP&{fJZ>^qZNH9<Z!W8IXakRAz-z"
        "I7-|4zljvikc@T=PP4%SupL9x{;+^+CaPm%T0bRB^NN$7((zC}$FCO08m7T-T4@4%W0T2Hxyj~$!UvV(LZC^>Afe|Kenkl}agExE"
        "LSw-<07_!-h*k$Uwt3V7s-Rg~;-pHlst$nYt{N0XVuV4h(cBIpK=kxvFKTZWK_{7bibe_P%9y_7p70V&=Tv0R81O5*N<Q+8iQ^<("
        "EOPXR1qi;vF;0W4Y6iWJ9>62d)Txx5I7S9TYH@I;gOaBH_19nB*9QXbU&Y<%L-q7S=6X+Gk3MA320#DPHuqV(7BMRD3GO8lPXK$Q"
        "O<kM-tkr+a*!Cl-7OWP@)jD5Sev%dI+Nbf=5*9qp$v>v2-c^~e3wKI+pbwcesrkBweqx<?#H^?cYcW^*y1!gUXD$1R<pgrc+OnRo"
        "H)y`Wl*ps)hyxN;k!KZAC`i-%hDT$zN^V8UBn>h;Ja|s)X_Dt&M;3{O!aym+Ob7B&^b~k^35uT}u1RCJ!;3aU!7DH^jGgu<GH6wV"
        "SPD)6|5WT2NK*-l;?f3+Jh@(I_OW#S`H)z!pkf7}gza&Tw8aVnZQKtYZD>2FqyNwVwWqCkjhh_)qtN}S!4=haDJm5xo1f*<DAuqk"
        "bb7UH`GRmn83Ww(N#x{o3|1%wDl>Gl6k-`vG<KO>C0UR%?X(>z1oT4X<*{Q*+@X5}?p#1nEDKgDE#J3i4j4W^PEtCXt@3n@_;^Lf"
        "!vqGqc@PH233D*M;8p>g5MFZ-4S+aIZ`bI7hS9oV@*E!8jhI<Xr{^0S`<XK}5%I90^K@Onlss6WL2^{c(H?0whia4r^@{-ZJ@Uo&"
        "RFXU7udz~KGLh?xw5oXeRJEo#9_#pojC=;MX;7&Xk3N`RrE6eVSFMv|gI67RJmL}aX(uILexaROpYk9=^+dk@D-rWPZzE%>SdoWB"
        "$k7!Zd4}x05Jy#va)r)$dw2~l4===_#>J)Lwco%C>pa4t^fQsB^u%~xad4-Q!1_a47~&>E)b9D3CY}0u(#u`zfzou(I8HK%)=>s>"
        "0}zu=L5q7kR{JHAU$-dkon4ifW24}Vu?ROYj1in-IATK9<sr3JE%TrW5Iue9>0?z@4Rk-C!wJ#w$SB_^1p=T=l|rr&VM&3ZLjypc"
        "XX_Om*Sol8Gq{q$qyxYXMUqp)Z4^^RNpO{6R2^0tA2nND+D?`<R-W11Oo8W2x|$l@XEVKSMkGz3hc#z_wCPlK4TO6wGGjT-OfPuS"
        "W(T*)T4p5h5I~QL?3K@%-&{Dz&CaFAv#7dB>ZR+T+8#Rf+T>S~T-Z*cq?+~7%E+W&qk1ouMf1=XD$US9BHI~3^8P?g(4!G2^Uw^e"
        "O9cvrKBM(;4P+PDZD6*p7Eo(!wSm1`qax6sC_ssVn|d*Rg@qtQG+~r32YLgYZoB@0{;@>=UB=SbT6ig<I&Ub|(<i8kZ(6^@Z~YD%"
        "Ur<~LYee`Fg!oF00u}5h5{k4~NQh}M*LjYf(iTJ%hF%qBkO)c+4$%~*u}H_G*Vo!=ZH2OS#kM@0oSZ&9Vy68QE{Jm}x0C!E*v{b-"
        "W~@^Yo_=J@tqkLuLBQcJ?rJOD<=}+dY}UbL!z5O9Ypnqt*g5qPMMmf>qHxV(=ab;V`-@LP;oe^!d_ttYLWT0x<^4cz6h8y26Vq6#"
        "TV1o1)v~OCG38Ffq7A1_#_dLa-ypW^{W+VsnTi@ak|t9Vpz@O6FGTj~5<{}?^ESU^Q41Sriw+kZZuV--?8CT*58jc2(b#m+J5piG"
        "XIga#Aw_9l<q|$xq``p&MURf3e(VFbm&nx)c!B@7>0PkKizFjAqO&#`1;ZbGO%_K4tF305<JvHj7<a&S1Nh?&G>+Xav{@ZK?{I=;"
        "E)6d#B@os^u@l%DSGq`JwgF?4j#L6LG?LYo8a^Aq&p_@KKho=kSWR=4FT}&k3oLcM9LTF`z5Q!WxmDqxKMbzbL|bQV+BeSKFO0Sh"
        "W`b=TQ}=mbK;6mN)UC`#U|$1wWkX48g4`4!+c65jfdtM>eHC%*!LGH7kg-;6^h13^&1@J;6MJM$A8Q+7Jp8_?n+$P%3~?#rK>={<"
        "2ba>NjMrHxYBc?i&?BPc-bMln3u>Az<i4JrdJKY;zJlHtYG7EPG597EfF4?J-eGVGLoS9m@cvN>o%T+p+I<wz&oVgFT?)POvrMss"
        "=eWU-GcYP`;RQtjF%nK-wF4qnRnXRewE_GyUdgz@#8%TNtI}J&kI7JuU#v$~K}BgxRJ#PUVMQuOLwZ{Dl(7k@Zah^STB+)&2={-`"
        "-4AQt&(=F)C8Tcf%fDA7ALe{?V`Qd|qOWf5<G<MMWHLctePY>8)p3ZzJvyQtSc?=Z)fJ0hBP?p^t<m7j%`2fKoia^WysV9GNg)}~"
        "##ClXuX}bV)6_~P#+`(<6!>P9G?Sjk?eCml<X{ca%s+H%P9MNPS*Mpdr4k2XIq|id`tEpX4K|`QnVGp)k@+mrb;WM9;(d8FfPF)W"
        "P3BJbx>ft?3W241fAmT?qrGbxcm<%cUqPUG_O-fy?d%-$2O%%%E>m;Rt>wr`hxoGmn4)$}kyWnzOGmxnI7YMS*LdmjqaT+6*;1Ht"
        "+DTOxIeDKeAPb6)u=`usmo>xFVOE|sH&P%jO3)gBRne|jiLBU!{iQRZQK;Sbu#=38yRUaraknUj$65v~t!gic$YqX3ojO6h2IMAs"
        "Lf~czak3ocBDus&O=9SPwsEGuH;>HviWVf%nxrr#3KewNwX{0b#*i&FlD2XU8b07DQ(8=VUk6PD)N|T&)Tq}p?1s*tv@}KDP2b=s"
        "rnjgM@`N1<CO}Td3&P6)@5%*ujgc}z*VS{GQfI4mRja!xbq?Bn$xF|MkKn6z8S-#RpBIsY4u~W6>sJWLMkK0DdyDOmJCA!vUlIBO"
        "dYVz$K$T4cjyW08tZ{tz8VPdZ=(BgW9j?K(?-I$oTP-Z!-`afA4_k_!@+!FIppPr8$pc%9%0Y<`V8TniaNLHJ_YudeT;1CTdd$*b"
        "wTJ>yvu1N6k<gTPr;$t3^&}2dMQ<Ri-9upE831Cd;MScY9oIk(UA)=Tjw(?}4S1%VRZv)^{iaFfA&^|WM#6|=1z{jho^d2eb@+<p"
        "mPny0Y@Uh&C|UzRbPyz;oiMOY;qlvA?s3&_GO9^|wXBYTSvKynI3#MdeP@H|@CzTbXJ4+Xf=w|GI47<HlQ?7xMml3kY7JB-iX#rN"
        "jdDtgCh%LwjnK~qak{Zii;vnb!(DY+J55<<p&opt?0~hjHMh)aMu3jibYM!52`lI#z$9%92&B_Nv-xH#Vud~t?>Hc;W5cR4my154"
        "_EhU}-^zlr>2#SZfE~X<KfDr`isXjKGWX(++No9}Q1%9082IU}_zH9CtDwMYMA03~ubmOX3*~*FZ<TqrhX_zn3JjJMp_dm}bVrea"
        "+dJlzF%y8kL6nY!6*tS;sFw%Sx^^d{N6dzK?fzUKGhmj28dfEGjq$AoE9v$43rza){$(@ccB$y<^*h>*5bhG5MvUIXAp2nyLmfbQ"
        "dv)#?cjT08zH)&mP$zMQf&*=yjDWktEB-**`0_Tb&Ku$p5Yo7&JJQ&L?jceqqO<Fi3&9pk&~zy`6oM!e#|s>Iel3i^fdr*1;O=y?"
        "{Q|Tz{=tG)r$>8%l*&Ac;_!~rkO0~R;dZ5>+!G=*Vh7btc(&q~xM8faOcB?AJ?+ByDY(`3Y<u!zzYE@PSOgWP*Ncdj7kd-cVk-m*"
        "B0Qg_^pu>+QVa54EJEHHUT4YgYfcJR7T?7s=YL?O71sNSl9%TS4KUDnHN17P`!5Co{TL@#N<GYfem_4w!nYClXYc2SXG6V6@!`7>"
        "w4*rfjVkJLzJ5|C$gfmR8#sz+tfET@SM^d95sc-3f5J`5sI04ZM<aHgJbA*7v}&|W9CB%8OkZ8ze}Lx*_#Ke^fOU?+R0l!#?zxz*"
        "YHGeh=l$tZfTmcVFrq9eebZw2z6O;iJi`RE)#t;EmduB_mlqRTX0R&<`ll-smOhfnr#sWh*W1GbYyvYc?<v0hGZg4A_yzEgxQ4FQ"
        "Vn^MI=|)5=z^v-N2Sj}wIfZlqa6>&cYZCSR3F<<*Z;6E7PvKRRJWg(@yYwZgS!TxQ@fMrfMsB<aTPItGMuT<HQI9vE)PQV8%Sd4*"
        "u|*9&aGN7Y!K!4KJ`r~m`U9vJNJx<z;~YL7l_vlrUJxPDxVGam+}jJajsAT5Tj8~K9^10^X2%zzV}tEOJqQ{@8T0YD9mBv^>P!_0"
        "QpbL3(;_jwnZN0ZM!yFJ38GVnTd`+Hd^Yv$F&Fmk+Dt?2tm$(y7=J#mt%sNgmIIGR<*nt75=_#?X@0ZuP)Y6*-4FSL|M@KSK=OfB"
        "Aka}WH4JMJo{a}8k3PsScHz^*RMm*r9(>$)o%Bu;-c9J;EgRDBP3rxVBYi!bo4Ubt&0&yGVd`fy=kIIGns(m-B4T-@q2bZS<YD9M"
        "qC_)TQv(^10cq1Dw-?msvMno<U5#CB&`|JJvyui2o*u|aE+}<^ixR&{&tkHpNpx7$6QV(*j2kc+>KD5c2<#``E`|BU%U$~QQvG}>"
        "e!iripr7F9)Kr}J9oO7`ETC<Xo&wR+@Iza+F_YiP^EcE~Spd>10NXTlE6|~_>b`)D8d@XXQHy58ZW!*4A~azwmbI(;EWtf>tkwlD"
        "KPDBAr0GHlTCXr$W=Op{X4W%jCJnVS0P&I#AJa2!-e24&8B^l7Z0>oLd0wlx(Fll9+pzh?GxXvgm1;7?tw}wJ_rR!}6Qxje-7jB;"
        "Ls&d#uITcNIUc@IGS!RhWG<s#$}kl)v)Dq2R%Xt+j<@z5e0$0b(tU<SLiQ+DVW(1Cf;<-OD=yEQNiLb?8mJO_!cN7z=d^PrtjQ(?"
        "Aj&53QVMF%C`d7|18;~g^86YcnE-aP^y~xUAXfFEJ|ZgFV5^KvBHg;l4gHL}BYN4@L(|E7D_S-5LlZr^hsCDiM<ycZK9h?q<VEXM"
        "QHkPn5U@4qji3sXL^7`|C)2CxNnsP8VfBtUH!!j*0(uo>Ni5PUx-t}RBor^CYy)dwzQ2cL(*HQ^+<u`sp+{_LF(In5cBibnO_Zks"
        "U%ap61c$PV4Ow@=^~B=?9DdmsnL&Z>>$m$ZjV>JRY+$JJL;=g;T&WZs^_+BqGZMS&qx-M<h}vc=_%Xd@B&%oJ*1hkm4>4Ld8KnmM"
        "@{9ikP)h>@6aWAK2mk;8Apm~(S)?`y006%n001Na0047xV=rQ9W^ZF*Y-eR{ZDB8DVRT_HWNBe%Zf|pRX=8IPaCz-nOLN;c5WZ(7"
        "{{f{BArpyq+)k%e>t;G>+Ucb;eO_b^1Cfx0ngmz?wB@+|@7)E!rzlzO<5tcj7RlYkK0oY&K@j}Hwq`CxaYq!RIa4I(^h!u=c&5J~"
        "G^dqeIjNL1GLyxO2zCQRa=qpmrv+Itk!?z<u8Gd1Vmc0j;Q6!X&v;o$Wk4q0Jb#w6HOOEURV;-+@(-FCs;_mJQ~1lwj2L;%1pmO)"
        "jJ%wYl5W#X3Ioce_Km*y<m~fLqs8-Q1g@%H6+D}PNPa^NBNt?u%;2AUyOxUJB7_U#<u8Nk&TOOz5--6@p~`5>+dTurfE=IRKXgw~"
        "VnxknMyyi8plA&SvOL6<mVtjBBr!Lv)L|6$7Q$=eyn8{`!KFAkIU(Qjf-T5BihZ~g!I0XkvBb9tFzLDs3ns!Ua=PI9R{e;)BKX3I"
        "kA_k#MGQV<A`d-DYeLi`Ue`qds{}#(yX3Hnl{_?`i=1sS{46gRF3&_=G6mlE9&m4Tkc=(YvQ`<!XymD-<ABhBkwKEgkV-OVVNjd("
        "{Ig(W%Zh;(D8-3eE=a?!4XB&0?hHdAIXfUku{Be$Oi#{~6w(O?&mAY)@`KnWawFq>n4A?OR=&0YSUM34FUm2r8b=BF=%O=V<YQ|b"
        "<(iS-Xi>9wN=X$4KY_!4{JIH%uBad-*CjRCCLZh0g*K|r43`3|2zSMUG^03FKcOPWSTbse*)YPkG&5j<aLhT=nc^z|jTI~8?F>jw"
        "tppcW#KH`R>Ow!pJ4uh%tJd>QPK7dqWJnn8DXjqO>L>O`&9njO`-H)BsKK4(J>HO>mv`bc;TMcj3tmkQdzD@_LT7-ad8#XhpxGeW"
        "FREryv@7we3HTd1;3?tHWLXsqP}0{f+h*`X!`})(l$+M@Rh?fkvzr-fBBbe@5<{Z!DtwX-tgOtPS72v>f~`#oImB-)O7N>q?-rf3"
        "OO7~-C9|-#lEfNJF$EV)ht~{JQ{>*gqxwhL4exZBk;T!{PRKj&!Wlv8IW72=fW!xfw#D=X!Ho`TM&Ck0a1Diq@CR>bfz==y_s2uw"
        "h`Bij9TBa7N>OpeLP!j|51<dCnNuBYigxxJAGyA7v2I(DUy@eN1Seh@(DU4FuSmC?zb5lmK++}mH2ZY5bU{2=HO<mbs+;Ee^5W3~"
        "YOZ~@Tl-r)Xw{h}W`&-@<IdE6dVfoSdYAZsjHCF$+k?WHh_y?UBZA!+{>q75wjuvN&<6uOLawJr`Zxg|kH?j5F}f#U&l)Re^RFj`"
        "?g`?*zrW%2)l}#JUOqnGrQa93J*J$HpN*`D?@k1RnM1?oCTo#XXkcKfaWl;Cby*Q<r<sx)oN_GW$CJ)!YY7xTKRY`+UL~F*+^@NB"
        "{@on1Slrug_yzv|B6)y&->SS2v}EvP)SS49C#DtdMiY^`pVlY9whr}9c^)i!y3F>_yIv7U19n4;s;Kp;o7aNF95B`0aE<L1u%4ft"
        "e=<M&bbkKvUM?kzc9p$ExO1i@<Qq~p4JNE)`_Kb?y<QfL?(X(6_%sCKoy0!OkyaAIa)<i)Q91pNl_gDGFO?Q-%QJbUXtm*4iusk^"
        "!Xzbcb$au9ul~(+oi<}P$hTlH!SFDAG@bz6fc)R*&EE}G+js$A`}=4!-DojB4%j_%$duF?fy<vdC<ws6f8-!<ZKDRVd+1=7#<o`8"
        "#*h(VU>~eu_-|GVX5{`MaxeXT&~K4@+mn@LrxzS<KqSnU3fg<qpO9?VH-XuYL-pk3F+#fC7XK}3^%hom2Z06;RJF0Nx`ycwuvQWc"
        "#tJ}V1!+MsF9s9903V@JLu*#a48^mXAX&QHf;>!mT!-**BB!5{&piyJCH)}9TxT59@-)l#<?<8?X3*d>l~!E9OpszKDw#2@;ReG$"
        "DdsU)9M+k|6F=VJqEie*EDYW@!=T$6x5m!mc;sMMiBY0!1ChYDx~^clv4?+b&9^a3*0$R06psX$CtTDvq%(5MVPmU}KN-MrA89=A"
        "q-q`N?g!fmpaqd=bB$mtEXa7+=->vPr(L>DE`BV=_JF6Keur@NpXT3_V>|#+tez`{-41vF&z>GN=rI%(Qk5{;egJ=4Ai;Pvu;`J="
        "4zT`Qz$2nRW>YN2Y1<w;g(`Ue{(bmn;hLa7tGhp~tT&hX<)sMYmv5p=0Th6M4yl0oXD<&A$wmiXer@|WMw6@X_0S{)N_V7*Zj6=J"
        "(84u7LA>s(-k<!jb8S3e&$iRv$#{tPsnJ&~=eD?X;}H+P_i8x^dgAs7dvJ`r4-uZ{A*H?T_qe+d`%R7K47lxdI@ogs-9qW(%l35b"
        "7N0QS2`cosTSeFw?7?pOf;{R2$FaJDms8}B&Esux+MLz}8mO(f2xf^!lY@5sDz1gR71LIc@4x8`gE68m+y{eB;HcI|bv`y|REh{?"
        "8qdOaDbYe}$e6{Rwx+h64zUfn+<Z8!VoSCh3rHezaj`=SR2{}7QBSAQN0KhK?s+mO$DXrrgI7H3H_py7+od`4ddqZh-F|RNZBb63"
        "c0gg6D^Vo8><X+;VddNR_+bUC>wC4Ar9R{X-X3X+pFfTkLQZ?y*kr|D_B??6AjQChl4}iHXxdA1zm6UAmu==eqO5;rn``^4|FYmP"
        "_tHjq@Pj1H2ZK9d=ytB*eE}o0!R;RSH?qiQWZ3Stjl(H4+M20T*s5$F(WH=-Pw)d$*XOy|>zMosrq>mKFQcUcS>?WO;Z~#}LUa4j"
        "9$|wb+Tx5i0W$9|P)h>@6aWAK2mk;8AplD$=9j1y002cr001HY0047xV=rQ9W^ZF*Y-eR{ZDB8DVRT_HW_4_AUvO`4Y%Xwl#XM_|"
        "+c=Wn1LQyGw0Br})|TJy$G{(Ea-F+iGMQ|W+1rI-2((1mO3RX>M?a>w|NB)HNr{r^?w(m3&Oj20k0M#`Dwbl~_S-6stvK>|Qt-fv"
        "lA_Ejzl_qv%F;9*T05Tjha=0ct#1{}SyDy`w^$NbInUA}D%1RaRHRkzLu(a8rEA;v>sPN|MaL}7OKbmIl+`a^i!`ZUj;uV?FD1*v"
        "DCYHhmB(?kb94T!;zjxTRhXwo%TME&)7}b8j(d|<Ny&3LAOnyvO?7^x6=dI`+V7)eA2OeNhpHZP2=UDe(%U3XSx^t(MPcf*c+ZbV"
        "<_4@}u3zPh@FHhAuU}d4`~1tVpWa+~pRV419*UMXX~2KE;d#umcS*qSWHZg@UdE!F2cAG#A^Z3pg7gUEq6M+x1<@W(FI#fX0*~c+"
        "dh5X-HM7X#sPz0aNdPkdH|Ktu2dcMBuX*BSd5YkmZE*5*X&A;pS+597b-<!zJJ>`BSmSw?M@e~dc6MJL(!>J}f!*A9G4ziQUw-(z"
        "_n+^6_^%JYeE#z8&1di9FYn*I`G-0}1#t4{H}0|E7?pU6Tb3uVtah5yJ8$?<txK2^2o|U!UiG4Y1NWBqNaG)`KD_zwpMUuDllS)B"
        "`>W5^+IkS1u@BXeCH#iPVRUC-sxwg6F53G*+i^W4WB#>0)J9ye#q?}73(w~4+~3XSyWR49I^CVkgC%3T>1x44Kb)VPw<ZbXjY}Z)"
        "Xo9~{8q&km%~%csP^#er`LFG>Bmz7|WoL&;mF4mH4NKmYSJ<mBGxsNF;rx6z4KL>B%WyuME<&HPi$$<ltrq^-MG#Jc`HHnBA%=g~"
        "&T?LS!%f+Lt@0f3J^pm{dGuF%2z#FJ^W|hW^)DvD6kc{`!F)Fh=l%s_EACG&{Fy&lgbc^xvIQ?)R2VkGe$+|!>!<fR#gMq)7%XZZ"
        "$}%f1$K!oef>67DdK?p(;;I;nQNU3#2hd~rZ!uPqKb}oyi_v5?n$6|dQPV;jareL3H39OpD3BQtzP@zMZe6i_Rq!0-vIHd1PS}zC"
        "mL{XZk3ggHvG3#5@nYhy&V$+R%x5!p&KbX01{ael^Uo)I#g>Z+J72KnJQx!tTFZLG{-8PCJKESIGWP3N0S~Q^m;S-4Rkw%YUpRT1"
        "mP6~ip->C!w=6FB;8JQA=mVUh%A|3FrrAAS2T|?-#o;K$`b%E%A!w|qD82NWUgcm^zXcNJQR=WXx32m9&~gx=D1|ps0UC5j+AU#6"
        "J^+Jbb!5>6&OR27+E@*E1E98Qi%lQ?I!e6LE0iZ}px><aVdx^2QQl{Kt+>%PjuJ;&*JtaG@tJ=>U7oKA$zx!a&;(0Rh5qjJ4w~0N"
        "2en`*+c-qwIDtYyp4<V<a&s~rjy(o}n+`Uk>9!dig;oLsags}vZUdv;a#XeqwC-Ary9KWIiYf%X=}?1sw*F$-s4dC}1ZB{M(YEp~"
        "yym>jBYwjjcpf%u3{I?#%0C8+P74lBFib{nfS^`c<%#HRgk@Jnai9rHghrbng5@a6D!51c8b*7PodPTk+oK&X59<$U(nt`T0!=%t"
        "2+ZPixO!myv(t=(l3NCJV;?O!u>Ngb!_sh`v#8+K-&kDnD{umG$Nm8vw=lZlnu~^L1`nq!;E98Wl>qCaVcc<G7;w<Qz81)(uHEG$"
        ";c6f&J!tB`QLX-DIROm?x`~|y;rT^y5%|+^>d$we#3!rsIS+!_ge_OI3usz~K4W|_54b;_E`!ziv}IMKk+2u@oqrKdR*MB+g={ik"
        "oH2iS9s+Q_n}=t+v)Oz(TP}Tew&M$cf99`d!E(v{T{vINf|&yPXkd82spdmD4Mz}yMQewp`zUUhLSfcBu;SqP2L?`2aA^Z~158Cm"
        "WE>!W*8y}|JfpwJiQqM%Eke%G*lk1T_hhFn#d8;@{&me|?eVhDBbUhDO6WfN$OgTP#CF>L&QNzPbZnv<1rTCT1y0w!(s8TFAo#T8"
        "lEO4_e2^Zv=6c<6u`CW&$eK=;Ez*Hnb`UlZOF?>;TveimYT``C?@+2ii&7Hxg3SO#Q#8b*Bc^y#9XWdNVpwM{Xp<2^2Gg^F@f#a_"
        "@MBimTi6D@?;sqx2Q0=F#uWB;Xc>nAHPR#khgTcA?XPhPx&4uZZLuYx6zZi}2!T`&dPnmjPQDK5xNW2m0=ihCt)T?CEC`&?et=Cr"
        "UOq(d=j)>_z{myAdQh<FrWuezu$0n2uthru`wQ#A6bKB~c2$O>b9-R1!U9<gVy-QX_8MqGJ{$Q1VuW&OP&l9)M{T6@t%Qqs0{SM6"
        "phO~i=&%67&`&ERkIvZ&Ht0<p1dkzwdC*}q@<A-=E#gvQmTa)M2L0B>tw5DPE`k&IrTN{(I0tMrtB7~d7d@U9+gW#avk@}@c|axL"
        "7?;Sx9YvH=n>6@e`nCns%LgLDqsc>(JE+%(2o<8{-F*pMY+YKanJ0m4+`3?a^sSwCtQzwF1r%0omqimOq3LAuq&0p*_ed^ospE#X"
        "0l(BFgW<T^7gD+Fq9GF`E$L_nAzKm#miq_r3AC*q`Ekc{04clNJugL@qe4%L{2@lkVvZ`t%lA`ncbqLL>}h#zMG4yfr~%`5We#Q$"
        "_vQ$w+Gtil3{<K~Q9({6`b-@y+(?|_QQQ%7sF>PS{GgFF@h%)3-6197k^EiIY)4))ly=};yU7;V9ZmNq>(@-~RH+%<GQ<=-AL+@i"
        "Lz<U>tYcKV*3S$AEh~&!35>~uE<{k@K?F;4sY9ev7!pm>3Pt?}n_wxHSnRlq_ElOHuzT<_-2RCQk|WO(i4M<m;5_yKJnfy6+oR27"
        "biqd9_Tg-}m^}VFT66e=^8in`K<~lecgbdpzm!~FBB_m`kjZ}SbC?vAd4-BXG$Q}OzN7tt@>V5q5bg9C*;JP2P$kzrdn{#v96kke"
        "D;7eR1>kSMH+jZ7nU6~gDtm>`$U`DX2|Q}39WC{HN{!4`tEGhu4q{9cB@btDHhrmbX{P`nFug9kH1`??6-uY8Z92N5t#oqM5h#kW"
        "Ya0yC`gY)w+lQESdRDjzV;}}jKr1H@k^xW*EBvzMN771^BthcP_5rbV1BofUYd}$JbO9q-2MUom{9o45a7ryGl3pJ`ewmd=9;7A+"
        "T%R!XUq$Yt$H3zBM?h;B!R{zuW9Hg6k}0n%uuMt$PL}|y4EF$|*Zes#Kulad;DP=e1%KWuUk$-&K0wcyA(JRS8pzT3T6x*z>hMQI"
        "7g#SK2ow)`blB!_jj>5yU`hcnc_76{pV^9ka8AUv@v^ryqYmqe1Dh>ZF!<s6vK2Vq3`D#%v~G}UBoxXWd(=TCoaNpT?SWSw03U}m"
        "4gj7&ZQY{JfJ0^+<6wRVTNqR3XM|v^R78(fiar`8Dg!ObN8CuSG-SspzSpj1z6WWsGSpBk!W24+NWqWT>68Bi6B2Qc;d4AZ1Q!%m"
        "s8G6<DWS~|72gY6BGc)4*<?`Y;-^I^al1i0*h`{xp@GfV19pm5HpP}VQ*;N>BrdGN&pA&3G4cKPR*|p_u!`vp;Qfqju{1bQRsh{%"
        "uzTV{N*t=F;?t3Eaf_E7cMt@DIl`Bm6VAPxSt}N<<7s*M<Y+Q3FVar*s?fvGRp6;wZlBs;8ivqnibK0`IANOt+Rdb|IBHR4w`huM"
        ">!f~Ves9tYA?l1qMa3-HS8UG_M>2LA43v8zG<9M<s4g00KN{c?I(iBkf?ZERHK$}{G-A-asCA!8pBMnb;Z$s=(kiI>p{Y(RQhL-L"
        "*_5d!hJEp28WTeskuq&Mn6z;$jtzbo4bZHgq{Ow<=Hf0EkZ`?q3*>vpumJvu*9E8V03)7oVTnX2IP{RGAZ<qvmI4LG8!tltne2xd"
        "0VK1@9)ya=L1BVbM0-$1Rn95()xB<X)V%>^E3`e=*%I3Lw(HeOls3STbRb_Z-gr=DG3rI`VPt{aZrVFQB$HXMV;H3L!KN)lq7_jN"
        "{$m0m1_+T!CTV-9G&cD6sS!%D1p;{-?IU~;`5jLwNl&io?HR?(eOUa?=UHhv4JwA#PyC*qD7fVKWWZwM1$-5WI?T4=iD`oIWI!LK"
        "_lbiYq2^%TbYT65HN#s*%x6su%c+UZgWa<^U`{EYZPyM4I!BIp8$7m@F<AqZ0DNe#W1im$QEaJw(I6}I)>G7bZ&JrQp}``d)E!BF"
        "NbQ;EUP)Wk_qGM{TOn*vsHLY|0S^tvy$1)0;yzc9EO|f*6+l67!+rG58{lo{Kq9bvkrIO0rb-5>%OsA3D<E^&?TblO1VLG(T)v6K"
        "je2W&5TXgf=d4o25#P08z^On_tY?$Oc^h261N92!KSa-BHBmh+zeKiyyhrFRfnzF{QoTwoA|l8dW42jiUR1wy35Uu^LkSVF!{qm<"
        "FfFQgFy3Xh3PHr(7YbAzwGGtW8zzKZ8Y=Vb#@u~JZw-*1_;AAeYuhz)@6A_eTHs_P_zA*_X6?p=&Lzk_aK$w=-lD349I+b65;oQ="
        "6>jk@20y*7P*A5*H6@x;xi%N)Xb0Fhq*ZFd4s_R1D*@VSXNa`oA3}m2ovamB&HW<QSqqFeLny`|xW6AWc&F1v3bC|skk(K6PTKHV"
        "D%3-RUlKrReaNk+P<P#~Leh}hu2rJhEfpZdsoT;P@zz^^o*dPB(^1|Q{`Tt&4o&#$tAu`28%ZK2^y9SP%3C8P4Us&*8H$=N^oY#0"
        "U5>)>Q$e<5VXy)H0yS-b*ETU2{BBy;rh?J4zSi1q8;??3q7@Q1V&4R)uu4~fv;uM^lrXM-ZteRXV%R&_AKNEs0a0xM#s-YoJs$O$"
        "N;3ig4kpmWPce<_Noca(4AIr~1Bft~M0mDJ&7q+^V+<sH*>o7FPi|q7L<qOn)bceLK{ml9@ZZ)XW%wbKNeo8=<ogE%I_>V6`F?e*"
        "Vv!lRv=-g|9XSgH9LF@f$}r_!2v2|(fl*lQ>H0j19ZV4kRDoJr^qrR~<M|D^<bm*9y!z~bLnq$|TzCLwU&8LU==bK}R@;Ihb7V*}"
        "B{ax@>puebDH0t)&TGD^S<)`pkt<C{fFR0)(B$rWYnOsIfGS6+v_erg_}+T=MW{_caSj4=U^fxYEx<=EEQlm2Pbe=Gd=^`bvS!xR"
        "?8oud;>T9C1*`5NP~_ku=V?s&rVrrWVUa?P7aS-82af3>z4#Zc1+dPL0#mKTQu&S82F9|rP*^3tRC}@#)Kfc!RjJ&pon<<*Bnq+Y"
        "WUW`g3g&=FfRv$h^Q4w*lEO+8U-q?h$7kxWy<mM>4j2I!L>7QxMcPrr)kduZ8rC(95|WN%I(+i9WQdb(I^qiaXBfThKz(dpYimHx"
        "xk7E@ogV>ueS<73Wr4xLYbdz#jp#+(6^ah`hkRdQCg3CWz`+VXk20k4+Vg_c_q>5V6$4`MO~o9?9*wB@4FX-Nq(gTFZCuD8a1hRm"
        "EwEmU>5w2LSygzLq8dC6S4?{`;0CuqZ|`RM9Qa!HOb1ZaS?$_KBOG$7M#8CjYO`X5s8Ut&EKZdbuum+a$=_&1b)T>fBRO2#=tgrm"
        "e1#`C@HkuBS9h4(0RJMZcJN@y;zmmqPZpFxCS_*ba=fK0T)XAHoH!~afwhZL^EOn)Q%<+s5)}P|zzU}kaLR~5lS_ws@?Pgxc$bF<"
        "t!c5U1SJgAv$6<$r{LsK>oPc4j+PQUV=8$Ri@pWrd_|pd<xf1UFvo?<VuJ5B2k+WqGM)(N8w$6%i$fx}Z`x>C-SvLKJsZs-zI;(>"
        "ZWEW*3KHU&R&gk0+h|uob<9Nn&<f+KIOt4{Kc`t8qY4F5GO9}m5ZiCV8<N+XK<qs*)bZ2`2+a!cilz-T4sVtob{bpBfV#DoVy`=?"
        "7jSm3FxCX1+xoINNNL>Dhtapw+PZnit=)d74_)h4wRZ;Mhd-Ja>83xlo0<2~jw*kFobY#+0{4(>1Ac@>{*J6A;a2_83@fPj&HH!F"
        "eN6v@4k;nhI$iz!<JG5kKVN<L;(h*j_2%=bS)vYJV6OFv4vJXqf_OD<6|)H@mKEjnGSEq)dZb@qB#;`@X>QcY@75@9q~?%&5EH+*"
        "Zor>HZvZSr1=B!(1HS!33Lzw@n}pw5pa>vL1hO~!=EW*PZnn<If?J4Gzf_odPJz5siTvJub<%}sGR&Vg)cU9yYY+8``azqiod-;k"
        "?#UbO7x@-~ih-^2#d_##`oJLv+_d9hL&aV_yBmgq)Z&*!AqH_9`N<9XE2qA>={kY7xE2vG)csivO4fLhkcfJ&-uwjAHE98y71FfP"
        "Lv%S=29Ij>l}aO0O{6G|NcP-W4RmOal+D!Ac-=%N9Dk$rBQW(12EwF)9lq+pa9Nh|#8k5M{C6NEcs$(LQP4w!PNx$eg^20(Cm$i^"
        "$;8`9XfOMY1b@wUA~iRgKNP{=>5oP5*U#wcuO=Z-HyMKOhpM5zPNAx=%NjRIFl0d5h=htex?)mf90SarPVZ#Mr64&t+*P?Vnk*|h"
        "d*v&1ha2j}h4Smv&=nDR6sT)yv01}~lIYmzbGF0gDD4g`axvSL)BHB<+KrPW`;&DA!3Ai0{74K>MYk|u4}_bE$%B)50THajGx+54"
        "`at;cn5c0ZTF9aix(cS-?M|_O%tJJO&H3$pn(kvhCUZMljIL%s;tijG1A(e2SJLE~s{5V>v3Ldqav3p^U7uI@n2**!ySF0R80AIN"
        "P`m~_u+qctg<>$C?$;>!$^Zg>93qfFw<4XY3+)jDgXG>pjBKvA^l1(rP2Rva8Bp47mo<BViXkKLr6wlV=#E#kutSjwQi*h=y*Qn8"
        "stVXR*aR3G-@CR^<h2Zo`F0L}h8sHkHKtVelZB|5IAZ&vVWX&Rz<Qo|&z1i_O!&8eQO2qLzMRhfg<F6VI9N2O(!$S<@7~di(#tS^"
        "MM;v}CclZ*wE8APZ-f;|NqiiC{-3MSTO2m}LQcqZI{E2DF+SZCshb86;-_JlEF^AfM4K9wYUH(aQ@ZuQ!<QXVtJ*RxnDAXx6|_S+"
        "a`ZRjNW=NebB*VE-iv$UcwI$ebff;b`(ykR7Jo1Id`^Dss#f^_c6l^QqW!-WjYSeSoH>5YV!o!*t(m%DJ6Y#=np~Zo_Ou<12V>H#"
        "yfK%t_NqL(vz{rN?ahBmo9X$Unlc(lQE4RpzmI<6r@f9I5sK@+r!qmkJ(!$Tm0lo6)sVrbOFmUaMt4t&=z9D1QL70sB+Wr9dmf5|"
        "=TRYr=V1!n^XyAyTQQ;j`qlpeP)h>@6aWAK2mk;8AppME8Bu=?000jz001Na0047xV=rQ9W^ZF*Y-eR{ZDB8DVRT_HZ)Rp}X>Mg-"
        "b7pidaCyBNU31&G@jWy74=5ixWFpaadLOOsbPvt><}%Y>Gk2N3C>jn#LKe>y$pEBf*W>@*-334b6e%mYPW@m>+{I#dvEN`21n*lZ"
        "Sj}@OviUYI1QS`_<mC;krKq{&O<t8u?Kr^M=7DXhX2(>+%ZwLQDcEm6ePCtPh)q@9syGOOmoHwv$oF+68<yqUZIN%f-#e~$?(g4K"
        "RdzolkgWFoPx11_R#tnKRz)FF<U%n6dt0?-Bcws7d4p6f5aoGasr>#uJV$0;zR1MZN|i{FRx%4EKP-U1w3)G2Gp4FmreXnd$%xH="
        "Ai(0~3kHAr7C1Y^H=+pxy*6ncY7s<ixn%I9hOL@PJ|w_Kx=Ze^0<#eRq-a`MG6(CJjYy+!#I}-R#`qQ_W6OTis>qVG0(r|u1<|Yr"
        "l;Yq347VkS(2}-X8qQ;297NIrlL*LB&C}a$6EIOI(XUeN=6BG1M|+D0&2oj*c$tdOAv@E$tE9&10Wo~b>o9V)!gJW>-@ItWwUm_%"
        "gMF?PYQ^JoYFDc%Z^Ir;3iP3R1^zPMKyE@i$&}dF%|lYjq%G4OFK+~%LnbU^Xm-PMLjRd&99Q&=dFQAXvOg_}>`AuTQnfW&I`AV!"
        "$@e1RS(ap?5KXsv%FC+EQ(gdvo<k55J%?exQ@13}Rl+x_D%wVbQI7yT2XP$0;X1ewAsXf5eEdi_HMyS)Sd2gKiE)*nquhs`COkx5"
        "K7_a*m=x}M$U&SS2DQPNG6amU79CN5vy>pB0-Fc%Xe~anteFL;?bSU4Tx8W=<fTxc@*-D_qgoRO<F?GVIV{274!C#!Sk*<|5S8#&"
        "h+34HT7HnNh-N-IS?}4{7$(uemfL_|JR-wS!2qSH48mpt0k?rY_sig8>B6kOI;Tj{E4Bg3$ma8#ykT}vW@!TjjgHDk!l-<b!_m4X"
        "^?GR#w)7dJjQZd|Vko14MF_9K@XC)L*)J6XhoVQkVSvu6EkoJ5O)=Lcn2LZ;laXS`Z*RW-CN!~yh+PKoL3)m0SM+>1_hvi)YY;{8"
        "PTXhtjet|O4p;5>AnFp`JM3IrKMly=yKRp{^#nH~QS=N>@(dCJ+L=72*lRa58Q`_${Iw2fuO0QgKH~9ce{=Z9T$s%F*zXwFIGE&F"
        "uyELPj48O0Ra*m8Gvz`$wgB3FpAE>nyy6~%M?SqqlWM%j+5iAxvWx>FTL6eoFylM>o7z)zks)+}Y3OP}@1J@M=yL&x{^Ej+HH~S9"
        "DvV&^Tc3VN#uo%06GfhiQi<UL`&ch&%iFR#l-<gwzL?Ubp%_#%3Mtv|PF+PcOH$;(^YZEv{vxlo(h&ANdvh}8{vx0N1(z-aR}E_p"
        "6|QAwx7eB@Pg7AtScbiYfT{#MA5IHn@@kFGz(8lVezdlVQ7A7naZik5ED~kA7f|{NS0Xnh0Oz!V{jlssSojfhBL%<hA2sq}fRHz0"
        "4~!j`gjR7&{o&*=agX9TdSP5KfnNr+jod3dUxG(#jHOpT7M72B*$Q7UU~A?0+Zr{3+ML)T$XdulDQGLvPLpU)Id4NG976#U9&PK0"
        "ZBI4kHD)=&^#e_R?pV8rhPS9US<02i2J0Ft7})aaTc9BBlC9+A!pqC}Y6fS=^U_%G8!%vv)BSer8B4xPuw|D9{JcVs+UI5HP;!Vw"
        "h;q{U-<5Z~$TKa*h2s0Vz{Ce@d)ueriKecoG6^5GbSo28*JSAW86K&T{i&}83sHueNc08nm_gB#muX`&ieU#g<!mR4Y+kjE7NBEU"
        "H+}K1^bz6-M3AdmD3^rg#O7CUSr83%=(0Iya6Ef{AHoMb0^4w*ue@`JB0~%x_R4~opnt3uw2L)7Iw}ktQY+5U8AYyMue#-N$gWol"
        "=b+c!u4hpjQtGT^?OB;-&vZC|bMmyVv6s<HE|3rZ6u43Xs%InwMl-M%<cAhMu|Km>C=h(|y^-+AzK<L5J%=h7D}_N=HA`2$7>|DJ"
        "c=c-V0RL{q!vYTeu!1k^V=(dv5acpEhT@;s)XyexbR34LaR=>ORZioTyjHuaNp!t6iAQbwJ)j9Q+8D!(jvv9%8)-mfAN(}?zi7)X"
        ";-;k21~LWhBm5M`Q6)D`s+k6wLTdU5sF<1p>)L4wY73?3Pt84+r~WJXPIYjCfQGG|gw^m(A(AcljF7d=%O<=-j~@8TWQC8zwK!|{"
        "wbJWmlok>=U)>~P!lU_KCDk$vX2A?Qbf}I%vKEuLO7ndA6Ne7#yt3aietOzvxDSHhZ3P?c1*^&eLKBt6^Q{zOZoETB5u8Mdd+fCp"
        "^|7qk3EILIp;8YWk*GJTtL#K2AUeV6Sr4j+MxBP&B;_T<!$e@Am{?An*-k$tGj@hGc3e~ktSj+c-PY$*D<x36kV4HrxmFXDZ7qun"
        "*bGXV%(us2li#4$FLo`&Oym&9bRGshELwGe3i-q>?CGSe8#v6}p3B?BgtgP6x-&j=GPADF58aK%hK;ik1Vd#U==Dby!yl>fq6nR_"
        "+P#^nK@Gv~ZOcU-c2AM)gapxfV@b!Nt$M&Op`uY^1~C%c+2d2Bfj2lNY?a7x4y#BUa<P+qIQv`Tbp<i#A;EJq8bjBy?AScy&L@v@"
        "6B?|iS1ICcTNHcLOellZ<@|4azFj|l^VJhz+x4HVvTkFF<xbCKQ;nZgy8O|{=q?;H$J>TEdC;B>X0KKauxI2ChXF$AaM0bHC0Z@3"
        "J5k~W167iBd2>o<9u}NGTy!nZbByOlS)C1Z!z^v`HkOahj{e9@YK-9V80gh2Y;cbUF(XDEj1Fe32ail<g1{&IznvKU(V<D&;zS_>"
        "*R6`gsEH=d#v1gaW0f0Kq8ZV+Dz2ahhM=&4Eus70LEJ0c_@O7-zz8~5j&qKJ?>8A=7}Qt_o`qMJZ@z;52tU9fHi2i9Jv(t*XE+BT"
        "g8rczgwYzHs5qohN8m8aA+h4=Iu*drRY}bu<n{lEl40<fBeHH!zGHVn=GzA*;PZiIxD?|}2Wv*PHX!#^!04r*xeFxrD+-Ks8`5*7"
        "VYWE|IMq)mhYCYla=`)%ju8my9yOVuHHPE^z{awh%y#jhTSVppOhbS*mB}XGRw*xT#D32`bq((kPX-BZp`~wh(N~YyJo}vVFeiGA"
        "l%0>a3wEa?_6)v&Rds}L5Tio6QoJRJT%u1Tr1RvYM4l9Og9KKI8<{r`-Ne9&zGx~-r|Z&5G4MF-CZV#?dAO6*tzCSg&fp2`&5~MK"
        "GV4K}*NMo)0n3Eb2cTX`0i`O)W3fP+iQsr)3`t5W9R9(nn}X-udVsWLIkCif3#BCFj!!M#>UWx}4mj_4G;Ih}QIK!#1jI?p<a|0Y"
        ">&h9DEp+l;fWT`<TuG>qV8|hgz{|j6@y<9BE+^yYb&K*Vqgnz#n~#07NA-2avcPmlRC*#5KRpI?AU*Qj(V$8)p|Rr##|aI1(G+?2"
        "SbAx9VP%r!+EWb`CqxYvuH?$AfSkz+;jDc{nf?ThE?BSV0G=;hJXPsQsJ?$`JVSN5QrdYa{}!Rz_<Kq^lxa)Q>z{AH&g2wH;03(d"
        "XN;Sn>VS1)*q^^$KgS+Zj^i^luRh*<l)*22bEIQ=x|yyH;}fDmbQmFue>6@;S%<-6lDsIK{%d4sr`R@P-$qVrMq{U`q0@6CC)-a>"
        "4P;DpI0e=P;uO@YtiN!abjE~mpw3}{!@uXUqUk{V-jfzWY&XFG0u(u~SF0g>)&-HiPV23}-ACUmgchA{M|6=d`>PA*r&H4XD)5yd"
        "D$>DwFS>dvgG#O$)@&FrFpT2YJ&-HdLF6~PM#Z*P`#Xm&nqI2dw+de%yAxQy?a69z5~MH@*?|$7nSPVa%#+xr&EO-}#NOgtJh&Gu"
        "RKoJ8m`RihDp7oO;K5FovKd|S;f)i719*;E19kuv#ht)iIf3t%^ap~TZsv8i&&L|f^qg1h`z0eNe!2R~)#bNu{4lJ6=oKdwe@?Yv"
        "uZ~bmI0_>HYpPpO4zG~JeGSJ4C`)gCWq5$r$LD6)Z5ieO&tuc;<IIUVxE@VN+}>LF{q@zGzMH}io;EGL+B99|nPC#jsN5%7UZ>tQ"
        "v>;7c*-t6cciG_mP?7YikB|$7#U7eEJ~JSE{jAN4Oq-c&Of2x}Zo!T0Sv^{|p1!kv(xc}2DR!0(TeGwYHC8m{?9(CN8PD!SdW&~?"
        "HNIG8x-MWSt)Ly_H{v_SWs`4V7iw6+eKz1N;9Fqs{aj?wmbnd247~7|0uNr2+D`5I*MGkI@ZtK$<n6zHyiWe{?(Ow2zg_=k0|NGS"
        "RK*@(Gt9s=KF}qTeq?S(&X~n%d^C^MIx?mO@f(~SUyU3gV#cLloSo_{z9wQ1!KeyKMSH8=F-e#H_=<s9=wIq-@vsON_JT}g-P>>k"
        "!TU`QN??52J^9>Wq7}v-tvjR}@;fqR0}O8zqK;+>pT&jAa>K&ZfNgQQ>USH~zkY9L{|x$YXk+efC(Uk1=J<tEkFXi8n!;yt`RP&X"
        "7)2V_eo+Z=Le)4v<{3n%Mz@Lo1yD-^1QY-O00;m803iUuN$^2S2LJ%}6951p0001Ua$_%IX=ZO@VQgn*ZEayMWMOn+FK}UWV`yb_"
        "E^v9xSX*z~I2L{nkpIBp?t|3Wa@xTR2A<Kjb7_kLZ4)%v1r~8ur6nrnL?Q)}$_118?{m(fWXrOXcHhP=Vv#&LcYge6&iR?QD);Q)"
        "=WnJzu(!&yM)-WiMAK9%7hW07;J<RF5?&U}%Emb5jlE}1i^i>tcM0eG@Zj)3)s3;98Ml74a{cJqWn)b)-DcIxx+zr!GNm;&8eUau"
        "$-3237*5z}Yvn0eyu%?HA<?BOO)jb{S=S=TjrLaL-t}*OTKo70v`+@(v(9Byba+t6k~vYz41>zhd~_qKR?eC8HfGa*;rIOTfWZe>"
        "S-R~`kdA|ZDp?es@S}}qw7QuxmXsvQWnu~GY`u))-DTl2^!6@_**EDP9Xv@mQ#uGY!g@DazEb{4fyIKB_h_C!vLN(*l&^#Z*IgrW"
        "a2schEoe4oKe2CLuo9+lxK`1{7SA`jtWZv}|KRFtYmJR~t6ke*(PWWrT8nDII|Wwy)~?;^>qcElaxS9hP0@VYRPq8w6P6^&WoI9H"
        "^emWyzPt0RHias?4tFzA6e>IwvXb=lBUt`j=Geqq)Y8qCN(*~GtCYU(TE4407F#)`ofjIiL+(t1JCGDyrKtpFR1-a&5ZDr);?LI!"
        "01H+oP70f^1}$d!|Dw0=&RK9Pn0WXl3~*OD_7FU054maX*@IKHs)SYk9%@v`yNBc%Btf$1`HN%tJUss%zv7rrHbz1$1EVteF!_HA"
        "8){`5X(`)%Tq`LGX&vk;&{~2^qN*Z@3hGj_=@j~yiCMHbJBrydeE*YAI*-Vac&h@Px+lG=p@Wsa%0PF9X0q^7?-8{V_;Ya?#@N%Q"
        "Dke<JTkPvy6JQ$zaHoREVE${m9c#nmtz55M0>CSE5sh#OjKuNAsm97u-GN_NUT|zXfzfn2eOMgfznDeIZr-*7ASG$aIQ0I_Ot)3V"
        "N0A`HDm_F6dDp<!fb#%e6Elu#f={Ff>~kh7h@LNJeA^gQ3Jt<$Kxn#E5L~Z}w#z7Rl)ZQ{zl_5!3$W(CnCZ}Rp<bjYgVNWdT~$dP"
        "Iq4%(857>(R2ZT!IQpYWa^YkN6#(3$<geZMbM(%Wy;rp1M3tGb8LG^3Q&p<Ke%zoL-)`#7#)rDTdc~%T;wXAKpTJlAl0k)s!&k2!"
        "uk+)3p-WY(uSbYWINC+F0bblXQr{?RbS<@y2yzRc5CL)FPmj1D1?}<2^_<;M-mWL`0I4%B=@cXiD1ZYMU9{K3uHXP}8IS*5D@)p@"
        "fb=2gWxt+$eDm&Y_VMKGzh~#KKcxIQgg#h!`~KvYpWmlo3VIq_I;@_*{`BG9$DiM4r=LzfoSvurcoZ|3*@TK5sKesSo2=(MQU5UC"
        "!z{W2Y)+$Hkqx8RKEcKyG};Mef?uHpOUz^MMnY(R`8RCq-ovfV#^SE2Uve&yv%#75o3a1`kftaANy<BtH>KoEdCY_(=oq?Vpled1"
        "ii8w#VkJ<kn=NKV2?JZZIUB)9AsxV6;Jhh|NcoIj4XNL!e6UvsuOrE`5*V|(k<dD%Djr$epU$n6WzyxM**YXfej>qxU6|C@g|bl?"
        "xb)n%Kzrq#ay~QHG<-tEEl>iPE$_W_5e18Ik4mM^fgixJ<;&?0JRYdX4G?dQ=zqbHZ%zR7bNrhd03OkLliR9HO95v9PSZ{zsp;(s"
        "tPzb&wYe2m^ULj0<tv$AW38eP#v4>JP99XEt+4c)YeiyP9qp#LclC%x7hs;W^GmE*Fhs}H1)MKO5X+QbPIgr&-LyUp9B!18J<|DY"
        "kz&2~ECITtJNUUzg=hecU`MqcEmIzLrM#*oNTlB$j}~{^Go&jCAcp~bO)u_>;HNuU12&cd`lXmN4%zE$I1fjz1=t<t2`UdjRzcBu"
        "d@=v-`1o>H`EbV)eckXdn(Hn|w2O&C(XLADRb`eDe+K<I--)}waJl;$X?yf_doV<pTZuQ`{@w8=9B4;(_p0uIf#lZM@wM3Sgl~aU"
        "KL!u$4E9`dzwYf@@*_g{>D(WAJh-S%dJ$1{FcOdIzsE!T?3o@+{e$fu4$RzY$NWk;fHy&xGfHd0fH+)9|05(ER=l)YRNYYupa^2M"
        "0jXq*ixR+pv;X+`BE$5(ZyBn|;OeWdWHF4iqZGRYU`-$rJl!XoLuna~&$?E?BLe-RLdFCB1;OEu9a*d4`U;JJ*O&-zwEKRwfLrQ`"
        "uI`!0bJ@Uk3gWnBbicxzjh2XxV8u(CoQ0o$O29b4UFuwGAZ(`<G?E`&dg%t8(pPt@<qNye1M=D}ZP?uB&!71)t3<}TyC3i~8Uo1$"
        "F(9<)%|jqXFBk6FLPyE7m+?ZQk_qW1hI<xo)ibDs*7yjaeX)Q_-w5C-9lxWXht=ysB%R5XHEk0;-{$5!Ar3CC{7bjv{?5}=e@yh?"
        "8cL}l7y4YnM()gDp2M+40!IvIp5~NYE7H{A!Cz2I0|XQR000O8001EXiwZaA+XDaqI|~2+CjbBdb8=%ZVrgb?V_|G(Wo>O?FJxhK"
        "VJ~obbZBpGUvYI|Y-x0PE^v8`R#9);HV}RfkpDpNNnVw#c}?Mmpj(FkLt4ziw!$z7M!MKsB~l})*lLmgz9U6Sv@Eyti$(J8c=z3R"
        "cchfwtsbwGtU6-3-hw4Byb>k1LP@e0c1L(y3d^4=$cZUb10`8?((=P!TuULhq!fHBm9e5Q*Ob!Bi^~g9H%eQ=jUB%Cy|i)1tIrRQ"
        "cYm{A?(cqk__%t!f4^d%KK_3H{xew;ib~A*3s}@)8VaRLD4A3i7ym$Eaq)8FFT1=b;h6|&8r)`0XLm}nK9;OdCA$4nPM$>tlGm`f"
        "y}Tg!Lq{Lr1vG&J6s-lKG|{lv0zZvvbph{4qrrfF!42;Y0j-rbgv*jByF=aRUmvzJ+i-1wc|nZjK}qQMJ&=QV!ELb{qRaZ|mXErH"
        "1!6`D-q=<{8C>=HQPm)v6<{<LbVhl#=bfSXaYh!Sp4DAq#r+W8Ywu;iCIYh@5E|v-P(Wiz)?5G8MU|63dDTMr{sXbo2ASg5-gzJU"
        "<j!MM{C_WyPzMjzn|}9M4bWm*=im^SZ)4i0qT<G6LDs<vTcx`x920M9FotgdFEmyU%3*hLESqwhHx0-#I~H)iZ>L+HV$mOZOQtSf"
        "Gv=^nEI*_BMaUBMsF1!K85o2Cn1O#=kOk0rHN+)qkY0lf8V<@3<h(Le1_<@co)jQG=Vh>mj)t#uf{|Pa6mjdEN+o0}?v_%`2)`b0"
        "rS!#sCb=K#$bB$4G?gfXoyqBh7WqjA^V2{kzRvoE8A_``7=kU$H;Tkl%>|}nEVOK(LEqM5lk2lY7vx#1I<?!I(tvd%Qx1>q66(fw"
        "lQc8a)-~5%7AqzgNcc_tCD9MVTwfa+8D{HkekvQSW4o31oX}Ip0_DHGrpLT+E$IK=I7om0&OvPE*4GiELbdW_7G(jXJjK<Im}qtR"
        "fg|yfLG#InNUXAWfMr-zZ3(18BI&C*%&^yT*V)ZG!pmA%J1TJPsu*!Xg>lj#ksNapCsG3W#D$Thaa<hyWDO2dryLid?E5qX+abht"
        "P<=z_8TdEk&63=lRhneMOJl8MiPNDhTo&92m@yJ_lne(P(zR=6^TB&tYbUer+vsM4mi_LvdlRH5IeELlTW-dr_a2C^z+xVTF#0(i"
        "=e0>P{6R=wT@`BQUOJvONQs!Uu{zPn&(5lGBQvJ=2{0tU)fe!yE_qD^G~CSmZn=)gXtj611P>EXe!yka^VrbPU?zh2!dPz>3B~<e"
        "<!qCPT*kU`>5h2iGa@~qrq|sLMEFqy*rY~Js{Yi<l#;3X*bPqg1{RKO;y+MJ0|XQR000O8001EXey!_}!3_Wado2I}BLDyZb8=%Z"
        "Vrgb?V_|G(Wo>O?FJxhKVJ~uJaBp*IbZ>HbE^v9}TI-YBwh{mC8UGI~`-Ms*p6=~aMww(1JL735aeR*Z;d<pjBzR(vB544WPIm8q"
        "@9qL1!N)x^X}`3bu}=gRi^V>E*ahhO{=2Lwc;d;j7BTl^yyhDwJys;%Pnq&$!78~fmA7Gvi;U&jXQr~U@X}IvWt!$$!9BsN(z|@~"
        "A@qI!<%^duvQ1SA<*k`q=h>=x|EDaA=0))4r>s|%C}S?=%NMCAH(p%kIgfD~8M0OE8M?P10aK+gR0&fo&Y6^4T8gI0kW@^e1h)Gf"
        "yrO*6x60BoX88@@Y*-kVg%T`Q4X5|KVwre-$2L{Yr}$dR4Cday$-3{CMnPj-Q5<5;@04X6v!X0AAc?GMBh=H>lN{F^iH37zMn1?H"
        "RtMH<L=IlQ@Ze!pmG2|!i&WXnGS$5|nTmL^%|uyjcwt*LHEK(m7RgdhvQR5DPEc`|zx?+8hu6{j*B{?rzWwFrOK<6Y(e!?r-Kkn|"
        "f8hnbE9!yV7i!H_7SGzH%`Ek_893V>;u}~YEBwi{%QfO#mj6-K#G~OXcRYfAE|9HmqxKARE{c3}aQU~_vo}aE`+zdd4iD#l@uv>u"
        "1wT#?kI&9l{3yO)hwSKho*bPVuTJLkSI4I(C(!lk^mKmM8Kl=|S~|iAzTyQ~VSCh*sNk!tmL!j%nSDI!Fg=`~ygE+!JUyMK{4`;w"
        "C#M%DN9XfXc77J09fAz!hb%o$pCO6h@)Ifg*IEd0kc0QHFK54^`r<i%bvj=i#uxMC5MEa2$?@taJ%+WjGak<`;-h$elCqAf@6l^s"
        "zWA;Q%m94(GcT4OM9smasLhF{4@}-Jv?y626)R#MWy!*mN*KQ+m>M*I_O_V>z;Jp*nY)DOBF^iINe&UV*<{LT&7`c|L5hhRu5>gx"
        "&3P8{LUstbRFRBDR%v$Lzr3GIfW{^LF;TWx>#}ge??F<*Kh>GwNFyI4%2{z!vl}f2^!R!LU+X;KDXkgw2V63>4;l?jJ-~5TKd?@+"
        "SY1&Q{CC}*W#Y}g#b#p=uxX85$R-5Y$}O)1hZqX{wNjN_92@|;tm_p(-sZrc7#dlpL=!?MGy=b@3ViI(%N-ZN#ArVCiW0YlFs<`^"
        "g9;CXf5m2>=d+6_oLzr?;J$#DkvtO*_8~tEeef6EvS@uAJX~_`BeuR4q7;E&7qYGpEqM}YVF~1XBZN^4EYki0?qBYWr*U3DkL1eA"
        "yj#O&VDtjhTcx%Fz@nO79PV_D%SQ$yK2f}nb8ZE*^M@=+0=@9I!y3lSe?{-tUG@GE6+bC;Rsdm^M6L8zZovV7jbXZOY4$h;0UzQ1"
        "7Kw5fIMF?Yd3M@A$q&adBDHx7E)101-=LmwhTEdt74DX+xmoEEwZPmEK_HC*m?_~3XuChvl^%O-fh0mGMSsx(E|&&eTQtI0V!IFk"
        "#u&a9xsPkIgLw`T`>ul$KDpx%B}GlS(#TLqt5!MRM1Z7<z{K()Ky&%XGOv+|hPp9U)NY1r+;qn~aN5X9xRYjFbOcUDm^)bSckzJb"
        "c>sDh{lS)SYC`Gh)?mqma?vfCpCE3kwKUa>oEl-Ljy#jeqkgf0@mX;b4cs*2dD6HxCZ3RWIl|eAP{Kz+4I+%q((h+)uD_g5Pv-Z&"
        "74mVpY?zJcdMJ$6M58RH$7pA)%V?}!z~hS?aBej0@^F$%JwX>6x`FK^J`T6G(FsXMRW>%ACJ53FMSmllIK6sJ7BHO-Qk8Wxr(NsG"
        "fo3Iw#=}i=;M3OBZ;cAJF<lmae3x@W?<5pJ4vwT@XK9iv9XPm4KQwp+Tbz?>{C*~G4ic%m=X2<4!z3wY%_c>e^prhuHwTtPHmfUl"
        "<TGoV17QV6+kveGuW*C7UNoMDpoT;=s*OT?3w0(!!iQLkLPgOEy@PB(la&=O0!XvV1VCWv*D9U80_mYBUW32qytf@9_)rxL_%lji"
        "L_iwrN+T5b>=9t7Oa}F^L<Ts<EQ03`EyG;}h6iZCVL?Mkr&vo4*EP%!oPb7lQ;=M$r;`9<GMqXxt|KER1lzkwj9y|>zdyFNiiuA("
        "SMKsmrCaTRZ@IG+XqxI1`j2aq6>V`~l<HIXP7`A7t{o%Hz<;8ZE}#SY8_Dc5UmhO4>Mzw1nFvFus;fOJC1`9TpPzObJ*6N^1aiWr"
        ";2ZDP#n4ZgvS$jH;NX_A@B6=ilWj8O=vA<(mLj{!3YL4nz5grhZktW580_}mDlb>^8!)AZyOyq_@|;px#06<jX~<t7;G!riG8ILE"
        "<avteO{v^)n|fbiy#VnV6<qB~al7zVWtoGUy<xC$S6#><L<Tkr@i!I%kX~$gFzHctxKaY^yju~!!{jB4HynT?aioUy-9Uqp5=u2)"
        "D-l4f_NtSLgA8?!a<!*Tp{R?XYwIf?s}!hUXEXRQOETfpR;IniOmaQ#4g!>2_Rdv0y9GWmtLPx3tW{MjDt~(vD%)`A1rMA%J>Brs"
        "iwO1334z>)nt|sHamH_solxYIfg3`7?R#Ha=29hGh%n*EKQJ&cr8G2Wm>XX$j_331F6W7>Xxi3WZm=lV-R%59*BPedt&3hdK+d;e"
        "cN8vQ-K;<pp3diU-K?$LG)K^Z|IDV+ts87qxp-nS0#ntt83M`@Qku{bo6uW2UVh*O=8!X#Yj$*c7FZ4;0WG-vBwX{mB)j1V7ocTR"
        "2M<tJb8kZ>qzpL+*WdC;l|eJ<(@3mYQN|T<37*Fvo{!%Fj>*A}WQc}QI<j+V0T5e%F4dc|E)pZhWpj3{HD~f_fF=|1DF--(7T9v2"
        "(36N>N$ARIor4x0Zf+qSArkRiO6m~uNdBn2wFgQa!JZZJ_%vZ3=a9<i_}4<i?{w^V<BK))eW-^bNa!JxK0e^{9E`x{X`H~vVg&@^"
        "^ua(hc6<smS`$8s8~C)?L3;X+p+`?}CH6fTn|gCZF!B%D4Yc}cvh57qL>fzOxyaJJuMsA6swWJ`e|*uu@B4m-k6p)Dmr3OwR6oCq"
        "d8Hhs{?QN{35)KsDV%W0Ix`qq@Z^6$tpyO!5#5M{xtuA%wR|G~+}72hta2b+`G23g0@1Dltgak~R3)TK&^GU;lQ{1l(@7)1xHXAt"
        "#DI8}lNEK(hLF=Ronm8E2y@c#R#wE7!=5TOptz}bn-$5_EAe0o(^eR2d705e9Ka#opO)Ur#{|*GbP##jld~@?|8PD(J2`<l>F4?B"
        "sek_*tY}dI%9UFg!PWzS3%qMkZUpS3Ci(3<1M1%_y?fylU~3OWCcVMhP+u=`_YY7D!!>J?Z`)J>R}E7JWcZ2q?b1mue%DfQpQb#1"
        "RVR2ufYHvA;R@*vkFt{yca52v;oRu2WE3)K6CM5aj(Qf|EiUameU<eRRSikM6&Vu!NTg%XgwLP7AN2L(!R4C|UX5qrG!>N1d)UD#"
        "F9C);rej0c(s?31hVfSAT;ZKCXdbGo_!c&*@VOBIHp37_hs4>Q9Z6)tkX2B+CqehJHn2`MC6<$6sya*&+x)LRK2&vOn8}HATu}iP"
        "WF+okp#a5-B*GJgsF&@I=D51cQZ7d1F(j(I<pnspJrMiK`~}~MZ@E5y#}bIsm5xnAo^{t_XfYbhwR1$XHOckK{NfC+fib5hRnW=H"
        "l$aV;W7p!>ZTb2U4=)+GMp?w1jt-6f)tY;F6`QnQ8zABWh)51)J4~-jK#$c)X6skuE!t3cCS`HVUTm5Zs{3ahsrHY0k)v}G-7OBR"
        "l)6AbOns<$cyw+DOJFo4#~b&E-KGb2E@ywWXJ@ciGNB~;Suoh?pEtE89!%bRr0{TI)Z8IXIzEVB1Pl*bTVk>-U4DSAI(~aUVpBW^"
        "(&i+8X~k*>Ke|r3E#EM4>l*k|%>+XbX)L_rAQh+YFc3<|ztCvR-rRvunAi_D5CoSGf5hQdyEdrxLd|h}8`~6wI{22ucx=u~HAw$d"
        "=84DBl;gK6TODC23aU$AUK3(NhQT3UvqHBr8)Vkuw|wa1k)S8DdoA8M<u@wq6~%)wwmNQhcDG2@{6x8Hzb+vJF|exk6@F#nJ37{L"
        "Z@XVwxOYkmqw2QFr8Jj*hDg5+`<qV;x`ffl<%3V~v1lFH?Q0IiO}~Bb%WiM=7OtB(G{%(BZ#~f!*?O+F01!Do?h@XJvaXPab^7iU"
        "bjP$9MWN1m(H0MJvG*Z%z4~=zDN*B<IHmpV61EPG9-wyxV}0$pm~EzYK?5Ox>Acf=#p3|u;z+r>@?CfJufY`uPTU(nAr$o$fBguc"
        "(1^(+^SVsh_!=IymAk3gfQh9hSZ7H>A1r+NuLK3%N?R94h<Ezj3(BXR=lQ-0SZ55f7@DK=ZIg3(W=Y#7mvZ<`jvX-U=`O#$9Bb9k"
        "wsn2ZHI%kXW2z_f$p44T(a93uL1o1S)E1Z(R`m+t!c)6)rayAwcXR>t{kyl{xVs%F9b#CT^IV9uHb?B}Ji<WwT?xBYy?edcRqdF1"
        "2dZ6m-W)2U;-&fP^b8d`k`1)ir{iwc`WM%qvYKygv}XaKw(azBAHqX;I#y_}S32LvwqN5snLAOoj5s#{%I77l$Ze3Wwsl-9lVd-c"
        "&qY4>fL05m!Om@`^*(O(hOgbKdwk7Zyt|P&_uc;nP)h>@6aWAK2mk;8Appq4Zhdnc006{j001fg0047xV=rQ9W^ZF*Y-eR{ZDB8D"
        "VRT_Ha%FIDb7^#Ma(Q24Z*p*Tb1ras#XbFV+cvVl-%S4pKHmF*N+jA&)0>`K^<FM<dY(&@jO}#VQG6VVge2A!sq(|N>hu5pb{8K6"
        "Kv0s~d-HVHL?Rc9#bU8vSRlUdf6R)4rC!7C8}Iw9@!~33Ww)&M;v)6(EMY~>(y^D-b;~AYk#D_<ZOS@p%4+LfzP<9|bPX@FvIu?O"
        "|Kd+y{3%;+%Bt~JalOj3dH1_%Her%ytZ3A;U+S_@KPslaHgUDc^6qzA<$x+=RaI8@!;1advKnwMs&efmWuCL70di_jz`ZWpqG6Q)"
        "NaH3>^0=;<0)PSKal!<^CT@^!`SK(D9D5(zihYDp-{VglB!SMeMVZ9;lC9Tqn3P3R#Yv;kUMN1jg|Upa-<L(asaIv=LWA^$Nn6D@"
        "28k;8;!hrYKAv6taDH`l_9lA$;muj}?)>%H`^&S-F@O8IOxcH9R^{>Lyhz!-c$~oFsLqx}+`u5jLw1jNqfJ@n;tBFQO0y-fQaq_Q"
        "dDcWQtO6Ly(x_rdS*10+9Sss*E*5!KK!1xy(w>)b8qt6N2X*MxO+1I*7Fn@m)uzgdVNSqoB&%+!=x2meR0EoL!6KAKdN7P)+pNkW"
        "`n8Q=JX?#Ump{M1di7KE=KR(7?>}5#oxi?}K7Ihk{;aqM!V}foqKWULySOTV=2|jwy=k`AyQprV<7z8FvZCGq_aZ?9$n}ynJ2|MW"
        "E|by?yooD5U=AQ=B)ZF*Ra6U^5*=3zxL%iSl|YN~u2zqMTUiQg5Llr-69}8QzKM`9Y{SXK|IvE!UUAJJh4&I#uv3l#R{jNahzjH%"
        "?VLmor`?M|pTMv6I%~f13I@Unl1Hn=EAcB!+J@;01zSozf{0VL@S-RyKoBFr2NF$|FUH-^s#N2~&ok|Noag0TRIui*tZt@n<Gg0t"
        "+civ_P<YXN+pv0ia(o;e^ZzV3$zTEa)esymBz&!2j809$LnPOml4c3;`v@kzF7sO!j4VildT3(&wqSc>0r=iCM;mioqLKN~(TYYX"
        "fa$wFW3qKX^W>pa=}eIF2h9z63=ohOfW`-|k^ofLldxXJFTVOq5GX9qJ^z1v_>)@C<C;ZW;zl7WP@aO(C|t37j!0)QY7Ic#&B>7$"
        "i0%Mq2#Ymb{^n~9vV~Waq=Z%>yU*&T4)k$pApP01;7DxatY#h>hxcXk78Ff-h6XN>Hd(S!+GH@o9@*Mr8R}#if)%cB(yR(lRTND<"
        "y{cL;Jw$9&-q260Wy4U-fXV_^(8#ztX%~y^KJY;SS4|e@c7F?4zj1?YLD3g56j0^Cmo_&Df3<Zy#SeC5S@95lmK#<C?iSRukDf$*"
        "dUPb!Ot=I@?Hnxd`shK5#p4m5zoQ3RE*|~yF!m-}v$AcbFTWZ98hLRIVq0&(!(amgBSd8r=fKkAVGtU>Vt+{E_V4CCb+H2fE8o=t"
        "VOW9ic*&|QZoJc}A}d7gADkS&c<DX!@GsZAxCwqf^>QGCP>%LD0&(Z<z8g?N2Xu@dz7&3p^OlJ<AhD$7T|q9yyiLJ&M(jReEM=)J"
        "kggBsVR%<%4cj@SIWX*|{|fusC<ENX-Z`}4Kd!Dmo)I*7Z3B{{JF(%Xi+4280oW@phPyysbH@z&8-gM^qV5J}0cKCG;px<Sv16|K"
        "^P?<;njo<2;O(GWxrnnId`%E!a&m;*uH);ZUO3+Ysvc9-k}?8U5p*{E1upDw>`*cLW8$P-DR4)?Wr>%JEENtBycgNAB^5}mH3ycT"
        "rN(aQBkMJejk_2FCfWhdeGbz&Yhc>g<cPv{1MVgZBwx5yooZ14pOLX=`bUmQ(XdwxVvNGUK<-DPZ_wu>agICz{FBT8L<)j8;LK?p"
        "(VQ;J*OD4e`X#e#{@YT+HDd*teB}t#H`!(b#&#;>MEOE^Y46K|$xr|(G9Gaa&VyC1zxiPN>fh)&;QAAz)C3ew=bEkO;0%N1?7!#I"
        "R5dsZBkl}%tFVaIY($2+wggW4C)il^Lhv|e@DvFS{39R@Ui)F_BMAgQ=%2V&5sorqbrNrwt%rdI;vD!Y$R`(C&RFVoc6{9jP>d#j"
        "rlL4B@F)J*I~mO;C)XjkRoy%}7*Mfr(dPLYgT=s~9Z&umPZrk?FOR|W3+%)IgylbV_PDj6Et$ha3>n*Dc5;l$4|cFrGRylf&Xp(4"
        "*1FA`Y?EUkST?l-yO5KtM$gejKIS&EqGe8TQ>}3oIGc+#HuLkWxT&MEii)zR*s{&z%D+aR<Kzi_=CsU!A0zKur_tKM3GO{Z;GD;e"
        "6l9wr$`KL+Lmk&6LV7()H=#TsWu$;c-gjDPbP$P3$73u=Vf!A3M~(s32`uRFmogJ<nVp<o+e32t;AZ|LkO-}|!U^hV-8I2&QMViP"
        "0WjBtynXWQN-DwxEaalh(_n-;3KjR$_?U1D3qDthQ%*Ufe$>Q@Fiz9J(W?#GfaO|a49;aK6p0KHdM+te3(PTAjcf?|5En~lJ9JLK"
        "x<{)-klcCKWDkPTlm40|FP%3Xp;piuJZ(u(J2K?BJV<BuR)kZ0C}ljA^oJ><VB2q-#pG)ngAk%mMTS6c2z0M7fs%*-+t*JQBLDL-"
        "SvDkbR}b^703Y;;_-RdtF<>atz)bQgIT};fQig=06n0G*9$Pi^r-<hl$ybt6E%_=Ot$s>5;C__!o|+c)zR*eF%RO}6`I|D&lC6L0"
        "`6$OXuwXx3z5O~8YMFP7`ThvKy)20?-(Kl~vrhno8E|x(&j1mKNkIY>DNP+fOa#12v!n^cPXouIEhxx|^C(%d<OZT5w3jffhCw?$"
        "qxYzLnMt`UP^QHeMnLl+*j;bZdvfW+n&CPKD*y81o1Z4{IPBz{4Dljk)#>N@Sup$8=i>VL$QP5U<`mIr6L^bLhsziqhs&yLH^B)y"
        "$yK%iXSwGth$-Z)0Gs$y3}yK}1|#)A!`>7S!}uH#VTC@CFtZE580fRjDzQz2nrOu;Hs5+nh}at3-C0zOdF8@OR+ylsr9;THJZh1`"
        "e6E;Rmy6~ut|%F`smj|7O_S9?_W{H(eGo@qQRc^+7-m!MFLmvCKM*qzr7yzc?v(>4$1*`M=b$T=X2}0}OPeW(yRC`G+{Drot^?33"
        "U)MWE4#WMOL$@4Ce_p4PFDFSJ11)|}bp?LGzzYRdAh*$V*SLdDJ7}G<f>FSCAYxrM8i1{JNn)_S7ybVB%1Kiqw@{aCwh(TWX|{mB"
        "fgMNCwPGyu4BiZKFsqY8#00@QF(K6Of09p{zJe7=ixvy>qjEk#>oTz0Dc}d9k#C!J4zlH?<vPxaoifxGQV0#rs;#q5r%;kURVoE1"
        "XDsdCm@9NSCg1npmYDwqvT?T!+GLn1ooJgK7nfIZ1;CpmZ&Nn*_?`}UEMA6Ni6SK6dm_XZu@Qzr;YMsm;`4rFi*aVuC1<}jQt%5B"
        "tpU%fZ;Aw&ksvclwn=V;e#(=g;8bU`P{b)ITw5zk55}NOYXgprnhBdoK2dxUx+Y_Lqd@v6do}|w7glVowYrY08y==hEY7HpB6<bt"
        "2q1k}shcWe4SE}nOr5_hT5rh;RzXl9WLJvi5I6`M_?HK^(g?N#G%NQhX1M0|agh~ioabO5|MfZi9D2duP6fcfAt=X=#m~_xn9AqD"
        "=ga3KFq!}DYwH5g{9rN~y#?71Ij<+XX-Z(}XPB<Sw1E`-umFusfeUD&z)X~=3<)~cM`}zEK<jS375}JSOk%sv(QAYGb;WcR!G>l)"
        "A6eoMT4JUiTpL=4w^I`TR9TjqiqbQ3$uhK4MZ$tED@U=Arm0?|W2IEl9CxB(kk*<6TbN3*=MM6~(D4N-@-4ayO9+-cu}j}+DVc_Q"
        "pv)MyJ*8)%<OyQN2c&@nVIj48<QOE`S}qUU@EvrXwE}{)X%KIw_O$Zn+MEJs0g8L^@J0Xt+pVp>L7$N931G2gK+P%T9~D{v9BD}|"
        "w8=XgS_v^B#C9rBj7rG3*@W*3;Z~m`_QBwmTt?p~AGVQug(KVp@I#+_4f^xvm|*B*%Hu5|nz_?mrg<FLc)=+(7mHkAA*h#i?aHr-"
        ">9Mn#d|(6c<boXeatUMG8O1ETk$37yL&|mz;Nj#S5m)Gemi_T}D0UkrZ0?(b;MVCuU{!q(x(Pb#hrn?M{7{sKjz*x}3yzK{I7)}W"
        "?85cff9#M3G+QX}1Ib-u4cC+a3kd#cJC4p3Ind<*3Z*dkm!5`@$p}U8Ugu9{O+{Xk3d*Dq54n1?!@Om;0_bSFHt0qNO<F0jl81t2"
        "a(>6srj3OotSGddAV(gtKiCNRjIi9XY0Px2v{f=GWa4({0-tiDr=F?@>j8)^F>5lQUmqkA;WWq@Iqe4RbGL!gE9fIQBppB99H^F}"
        "494J@W0_c#YxvDPm1L0Nq9pr#lw>8RNubd<@!)gbMS=-vEoPh}*JFcy{Ed2Wtr3X>$Jf%G*h>%3A_tr{F~sX`P$a_l%`P-G6XmIz"
        "`}HHMAhh<dJZ{UKqBv<<aJ#o2G`gcm9D`c!^;Ji>cZ^k*-@qd9$|{CXjC(^`l(^`6A}7a3KP`fI9t2nBqA7Q_GXrT42sgnpSlObG"
        "F~Yb!QXR=9L9}#g@Zk=e#_~?i4so47;^C)3CVx?b>pQz1Jqm>JZJgy;GHQ5~`+Z2QUaOXAR}7H3|Lc7_?Jz%y4N4&r?MxP0TFKeN"
        "yl}n3vP<WFq+(C10e;8Qr7}A(!Ef4%Sg~Q%q->j{gfKoSt4&*blpx)dS<%$rcxlP_G}l<DiTN9rW94l#Twuw|_4pPx@TtLCS0ih{"
        "a}V{e82f!GQOGHmC5=`k2gPc_7Cogljj9>aIm!|5FBGN7b_W756y>CW|N7$W)zz7I_3H2M&b;%t-un+%-q}y*msgi+RDmjp_O8x;"
        "x*GXA@LB1*9oWTm)IqseKu_NJ`>V6>&n~==7w12`y7<}q=h@HWO}wSjSl(?<VG;p~wyC33I#4sCwZJJ(Wi6x5i_5cjXRog`ly5IS"
        "{Giz2kHQ55T~-#%%7qAw=>PJsV)I!c+G{{NVHTy!yrNxpjk#tRQyMx7EP0Ps551wCgQ35;LMDIcu>Sb!-6t^r!QU`l-_7!P<d|T-"
        "Nx}nXxfZZin%03X_o!(2l;Y~(^T$=!E<q8j7Y$`Q;=dFW@599#kdMFrETD9Gu!g_Q7zEmq3OqOwB0e-mgxDBG9p5ssnk~9}#mGY("
        "RU|5ojyX>a64G|P=}0(~$w^+;tjA^5g5GNBjCq^osb2Ta)t%rs*S)fq9j1>qrR1pw%4MM7MWA`a!=}7p=+RVBf^mjC?$ER;qAnKM"
        "Qal14Y+C7_>9DfGT$n*RB!H#ks^%M%GfzR}8H|4Vm3-b=5`6sf_^*E%OSX`07cX8Mzx-Oh$m$ze3b4%P&_=u%u58~_<r0i^J;lvt"
        "tm4;l(qOQE1OLL4RR$Bv_P|Q#JpD#^A?!ZMTe({li=;F_390C(qw~1I9TdTJtoq|`FnP6RO^jz7B(T`tkrLrNN&}N1!OwmIC;$A1"
        "v-elg<;SzvmqV@S_=K2S<Ma&G``x0Gm4U5Rvp<BV0bPbiIkck~`gs=FzJCxy4hYpJPnwA!<-xD-P>0{M%15vK4ja$G9au%TFZ{<n"
        "zDLDm<mHM#uzV*g0$mcvmiUcLu^XGhHnv3!%W6(`(v%gqah|1O9g3!sAc(rURR<=icm5|Z7o~EVmbj&aRp{OOnUABW8vlCc6TdLa"
        "jwg9=;}SezU&M#=yu^)WvHu}aK^y+Oo2m>W09pB$x68WMB&o`}M)oimqdvz3t{TJm<C+Kin%>~_M&hNUj;j&r9`Rq@?D5~O^90@b"
        "fHQij4C|sr_Y=bcEU%>UX7dmYPKz?Eq_|}7^h1Afuut}950CipBY;@yh!jAePPv4f#eBaq&arxBJCQr|sV)ncJH8La!0|g3U7Cm1"
        "^g}){n0U!+w5CM7B{*Gz-<JpGBErk+BPA_$$lXUh`$ILTI1oNu{z6IxdCODhX(!G->`$JOgyn4NVdIzFh|8)P>$(=`RfC|QJT9OM"
        "Td{$QB{KWfxc~A*y%ZA2%Zn6py~2q5ZCziz;v?0K_!1#DsyH1`mVq`1o`5!us5Jd0Ze2$Tz43E@CmqSVdl3q((`0)wca!hpy29t|"
        ")Q}I-uywy&*gF3Q0K1L{%2siLbx}NHCbWezz4~qdsg<Fw9YxL39iAE`KO#&$HGL-S6}e84${Oj_;L5Wyt8=;)sf8%mRMX-58$D&@"
        ">qbju!-Ngfz)WiffR=gFKn+6&kf!0&01Xp}2{OYZ>VT$kq+FWuL<d%9p(EkrV0o2hLuAq2?Mo>aX%;VwvTm}ZHma-ouOn5kweQR-"
        "N1&M{Yi?XIr3P)5aH+Mxm)L$(X4o<_*S=dL*y~u!`WnPkAQ;&i1=z}|`uuz?iiU7R<*8pW55dSvRr+*V1t9?d+DSO`dB2rphuNlk"
        "s<(LT>eBG9oDw;(BzL5FH0~)|!`sz99#7nr4xmGe!q|<v1dgG-@)+$p&rVFe$%%}Ty7*S6l(1xmIu0o$*8LI_3IC$eO<HQ^v-=nv"
        "Q*c!6>PqL4>cBoftwIe-3kua>HR4%9Gk;(-!z?Md5WXEut7;ViJyVd^CN`gY_I$^e!}iKh5CHigQHdtj9+K4ba*u#R^=i76nxAn1"
        "a7g2#EV4v8qu>f3U%LucCEsO7&Z<^7?@bu|^k#!ovwt$iYM|s>nV9u0lm0e%XZR$K6id}B->z(zqpsL5l!`sYlfbTa2ol%$Effbz"
        "PW_R^N<3&GVrPE+g(m3VO}%4{U<nsI_(mKE3}d1R?!<HmAnsE$!|P?`PnG8Wvv*0$n@)&%7;snUxupz(Xom3<UB1C1amxKCd7Q1W"
        "2m^FhidR>Zn>H6|wiq063}X8$yoxFD&I%wEu?F^C0h+D1YQ_XFzMbkLv#S|;Mbo0<M~xSGsoVNJ#XtM>iPTu&>u7sBeFJJ0S7fT7"
        "gO}b1z!xsxSpjOc0A^&VFB53Gmg$$n(}Yb2Z&C<^zlE%~<yWrm;luD_8Ep0s0T4_c7t0n8DXMIPu!du_i$;1ue)=Ij*4m3^{cbuU"
        "0lkd1L88=ib&AtJ!d@pnKeK@m^Fh6Lu|4rX3-BGM%QPLS2)IU>GTp+1I+jhuto_=SP2B6VyRN`tegj3rxPP*fKk0TVyIn(VKnnj;"
        "j=7{QmXyNkjp$~cNbaW!dpBV2WLMVvz7>SAN>~e(>ufB=%%QsVf$n=So7vl7{H}rBJHUP3*%lQyNhJnt!?Dl52V)ah>_NitDq?z%"
        "IDKE3pa%Aj@navAQJR3(1O8Oo>Qmo)_5O_~&T3iY+k$}?uuv{7yQlE*!0TTegC`pLkF5)1^p6bR`)`bd+En~<zzK~t6@NP*@N}Gv"
        "&d7hRr$fP-i_OjxNw-;@;ret)+u+uGAB({poQIhTE4!_3M}^S7b91{+;}7pSqd$YziCT_^Pt?!Z0yD*3;ias)G)pcA_sZpm<n16d"
        "1nI_sXA^KonAcUPx{3aZb`~gBbPW)dQtP84<1EfsV}PSkYQ7pefQyz45u<CzbCLXnFh1OML&h$9MfE`?Phh&Pq>%R*fIBM=UqfVy"
        "C&P<Wu}B-p02O^_29F>1OO1@4T~|--fG6H}l+2L2YhI^?#X-}B&u>!Eyraj{2ZVZb6o<;sK>3^XcoS!IifM3b#X&c}p~`;h$nHH!"
        "!O2YEU>mVRw@X)Pcm(uNcFDmCo*&66qlaFzkuQ;j-?~!5dMB{$5zcl721O`S%G~#zjSl^HW^(KxrDP^=Q0y0Zd1uSLtWV;I4dsi!"
        "pPS5q-(Q`)aAWvAGSL@-m%t&~3hn2O`Sh3s@0=d!L0+BVu#Tjyrq=8@tmC3#)zo(tq|#=G9#Tp?4p=KQFxA{)m+9uJRFy<b&5Hpo"
        "nX)rq`00*T3)gQdzqo)M^^?-*PVEP-!AX<53(t_jcP+>mv=>tjz?2RoYFo8fYTND7ilcK$GMR5DFT-OO`vqk;Y&-4c)8&Vx7nj=-"
        "?B8^tD%=+f%0ES~DB$-SWOA`rplIvSxLdidsUbAjC=!w_atwq=3Ol%vFgij9i%z{32ALjDyptj998AZnH5_WBR+dSYeC#64jp@1B"
        "PV3qY?;gmbOVD^765smZl^lG8yZjc{fZb`aY2S7`+KS65qeUJ=lbKAht>W6k2eAvvY(Kd{0z0!8elKLQl8~jHkLYDF=^B>1;WVWd"
        "g8xTzlD!OMf6T5_;P1g><t!byuZU!y!$zt+pj!-godf9=G`TkBdaXE1<%xOzYiRjV1IpBCsmCtwwX2n#+&%53gx!rDCFXk8*Lk5@"
        "=V4e#|H-zl$Gsa9t@npqw6^1dHF=-YRb1Jx2X9p4xn+?^<e}ODd1~HjI3KOCO;lq+MCP{}eo+W!7GYjgdRb3f8q@5_)c2|A@o@mX"
        "O1!HlL)t-0tm96%{Pmw-*s8~YO$Y40kDzNW_OPfR(%RxOcX}+D1nPo~efu26wG@~33qeNVGB4+W{|s}VpfbXYr_g&B$$V#TAbZce"
        "7tfx3sa+Il2cuGW>ZZi|F(M6tQ={5PdY_hmw|g^o>GyQ^gs%2<qk0iDsH#G#8;ahIM^v?kj{WVJ-&{>MaX4?V?#Sa!d5zf<Y9;>g"
        "-n_FWWT%exwHoS3m2aP_UEcrqS`KG<-l;0<b5=<>Hu4WKZ=Lo*!0v|)CilR)+39pA?XL0EsoFN6?0*y&Qcy@@1T#@e^&mj^B>Db-"
        "?HCQkla3dB6w2%0mrSrd`(5b-J{f(t`+yjH_%xR>T3S^odib|>-47McMw>i#%`NaaSs^!U*Bt0YXB%~_>n$wTb}cqG;r&bdJ&(sh"
        "qT4)=s1tyqFp4O`VCTz`cXZTmS+_9x5JWP%y)-K#HIu0StC?@aZ@7spq9DG5K-WF+K88fY3S(uU+DT-=Vi2+}F%5Vl7Gq_wh;3=%"
        "-kzA%K{68W@MP<CD|V0ZUM^vmXq&jZLTMr|!RHln31{i=D9N%fsoPr_MRySLv0yZ%vA7?90e8RiPC5=Nb#2zu$g_prKV`t@n}qvz"
        "EUWj0de^T<=x?PMCkpgRf8z+iA3IGT4-Ih6P3<yp9i36m{|65l=w}HIJ!J4gbuoODKweVL(<gL{$lHJ>_HY?`{L(O$GCC489m(Un"
        "*1sA;+{K-ZRl8h*3R%P;TdP+6D;%o@{!+L14j&yD_%{?TKYY4)eHLAu{dj(P{^9-Q|4FI-r|$AYy&Ri&5O${n7pxMyQZ`vL4lD1w"
        "5UF<n)URj)nkGcxgv5Pva&mkV;%S^*%fk;PwzOD`GF-^7BaR0p>v&Qw=*tNfSO40oBT3rPPP~iXoDPjTH99l#E{9?iVWWmCi)q*7"
        "<l1--lHF`Kuu??sPFGa4)^HV0wX$|}$X5#;jISp8(=78A%47G@qHeWnD6Na0ojPGyFAV4g{1~bRKi0_l`2xK+T}cdRZA)JaR?(XJ"
        "W5_u<Y=^_vEW;+ktfSWo^NDn+hvb)THvbn;O9KQH000080000X0RI=g%=ZWY0Hz!O03iSX0CRF<FJftCZ)0I>XJu_|VJ~E1bYU-Z"
        "cV%;JX>@2VaCyyHTW{Sq5`H(}|G?G0Nb7UtwtGS0;%s*pNxB6V-6C<@hr~UgCCZ15M5-he`!vXZ-<csLQI{MiT@<Jv5{sANa4z3`"
        "Lnle{<deU!S{J-#g;vh+!m&5JmL+#mt1p<+j&b(lho@WHNOxeRzE`#8r7%nz=7e?Zz>N~t=I3YUXXC<_yW+s4Ww+hkR;scqq`J%5"
        "^#O)TQOjFlxD)k}8IWOQ;Y10CnFh}O1%w~CDr;fS&%V2Uw*8znx|EeHNGZ#7#l(ZO4kWTl!m<s6<?M|xut>m>dQhO0MihS6J*xGU"
        "bf9RGB&e-yT5TMxsVY!t_&#uZh`)cbS`8mY48Pj$wl%tdyAMN7G%ad$epVUXFt8#oYHlqk)gRUmg(aXS9@bBP3!kVO&E3ihJ##0T"
        "hUdL|21(CF%cXhxz?-%f8~oH(I&F^6<n;T6;;lVshjP!)N>MQzM)r8swEwzV*^{yD-x)aQ>imqsPs@+!eYRsS(<(>ZH>@Jngo3pd"
        "#6-oUVipt><sdl^UV1H#yP7w*C1=$Y^8&IPFIrZt7Ak$)l9YDC_9!SE_%36A*wK6#$bv%nW4N?}eb4JoJT*p}G<njsHAagVZc28x"
        "d6M~U=moe0dNwb+rnSA?4HL?CM(nvQq})B@wFO&RFg1j$Hg=aLn`Fb1t0c>Ep$c7!H0fNm{k#`#gzJpzgDAHL{@Gvumg0d~eh?3("
        "yn}#GGw+Ol4B?Q1$A1xOcWpWWlQf2T_X4x&xhS+L!<D77j;liKWqF0J8E<5T+Z*vlT8Q^(3NrBS2ou#}Y={G|Wi)>(CW0E)c48`V"
        "PdOLHKHz%a^!Z+{UR3zdzM4k*Z*tvAK8W(f{BXJDlp*EbH#=e}4MNN%ex=c!DgpS~y;k*c4>-Hm6+oQi^&Vl>CS$tp@1)!Fwylq;"
        "$1`xo6+y*@jV9l)KW#u8KtT&OyVbge(?S3j2T{D94)z&TXh0W5DV@MM7~I<e{{>HooOezarml7s?!jTwgH~<t4-!;uc_DU6-$NNl"
        "#$4_d8PhZzl!E}F$`tm8HGSs5F0bdQJ+xxXHQUt=R3D)@H=Rls0@wxIZkya}AbZ-*?Z|5L`~@@Jm%G8j@X1?;SfD$Hn=p1a%*jT;"
        "!0ykV_*1gz%BW550RSbR#HeqnKKYvE3mdCAz$7Gf0_p}~)<LN?c`0H|!T1n$HP{_cXbh8~eD3YNpehVmfccdA|E^|uLIGm|DILWR"
        "LH#2s35D~ycN(TdsIj0q?~894$OI+afFfYEgHV3WF=Pz!1(f+jLv=)+AyoyH9}?FxDt|rDv9F&%?GO)uLxKs&s1Bi+$OnwWZ{k4|"
        "9e_RBDEC6Y0~BvtszOjk6*tFAUY3`@JcJXMMkwA0uRa`(b_WlEjvy772eaS@<$#<FCe4TY5AH^ly<zh?8lh6r^nggSae~ks=}((J"
        "QqcSXtUrLv0VPhlU}+SNaZVx`Hv&d}?jV%QJc%Qe&K=6~EePFXihvxbuXA-zs?=8l77J-QAcV36<DsL$ppkk|%^kA8Dcg+Io6?e*"
        "ve`nzIb5I8_Ft2Dr0UEXX&b_UNQJ4_JgS3-xN#QwW11jYWZNwuB}-2G?CC=*h}Lpsovc$nKTyDM;&25bNC0GSJt?{xS=C6Y=H8<{"
        "?}DJ7<)V#pO6t@k>6cBGe!2TGQaVdEiwg#23JCo>L6sojI367&qXKZC6DIveE8;}{#MlT2L7f5VK&Ay9qO<vB`trwD_U0mEx5KZm"
        "R5A$5Mp3~RK_ckke|!1q4aPzE`0OS+eBBHGh3X2n|GCrXioR{wqU~l~1t!f18Wg4VaqtxWpl8@_)MT8QlJC?L<0P5hhpQvZ7&nP0"
        "IKUHL7(yFkRj2(~jzbgkuHJAuO(6AeSwaf3MmSDHm$)z84^0{Yqfb$vkU)X3*fH4Bf|v7TDtLMovV_O%c?{ECU>0UVKt1XsUkYj0"
        "6hI)*#!hcYUh*A~!vYanLxIR=hOUxVYP;P+yuC%C?_Q~kiwg+3{wo<&_hI*_g8!VCj!R!th6)nE7E&E(2=uy?F@q(SVK6<gmP*7y"
        "ktVt>F**gqTCc?Y_z_B%Cur@Fe);O_>rmu~55%G|v`~C>TaOMnXwzu*qk-mA$%zN|(TNiRp{H>a-V?}OM#>7FHmwU5AB0m+-#6X)"
        "wZP68%<!b=aA^ybM?UP5xRu_Lj4dP{x3rn|L}w2a5R#qpaaa`H>5uFO1N4_6eGf#4^6U$ykRh>qnzgz!z+i~A0&9%f3h@^J6VsE6"
        "-e0HF2g2k|Am<&myreB_%Me2#m-6N{tWN!h4Mdg_Yp-ATJ=w}`y2Bn_o=&s;B_4Q#%`$K<NY35BOd(h?w00BlTWesJWoYPLf`VkU"
        "xpu!sRJ7eUdVRGmWih{5wAS;~s2qNngD57`xp%T$L+qy~4TkXuY*6s#YKiKR=HB|)_RwPJICz=PJQxyd&27;4l%e_n7G^yuT^ul5"
        "x1!TBN&geqk`WifCAR>F4FDD5gn!rz`9Y^nAY*huj~dmKN_a4jQGI<k1TXGhDj%Nx@<7b?_`R|We^#v+!RTutP(`0v*=AkH(!L)p"
        "0lFgGzpIk|{>gimIvp0?o>(|D*dp|OF5#O+lK-ToO8xKz1i$w^9rB1fq3BV!>@$Z-8a>itJBLf4W_OG3@T1^wC9!^Mlp0O(k(e}Y"
        "%zg&|f(=%T7s#U*F}bpj;$e~K{!I>&s2qAGXx9x;A8bv1<3rL%H~*T7@PE{R->c2nV)TQ}4t?xp$S$9p!#m=#o%QD5t-Ro-ai&k3"
        "N)WCis7RpGk#9t~C1Ej$xUPjxXSMN}{5@bY6#Z==3cVciw*fMm^}siMN`K#*RCJ5U3voKH1OBmPd^Vp!VS<0Px&<d6E{DHg&Eb-u"
        "db28cc;s3?79V65QxT*@41BY2%lu(!xng*%dEeCRl_Xj=eQqIt#461Gg0XlNnZ2G){<?8W%y{<?5dop_KTt~p1QY-O00;m803iU+"
        "WD>Kr2LJ%o6#xJw0001Ua$_%IX=ZO@VQgn*ZEayMWMOn+FLZBfWo~JDVRUJ4ZZ2?nrC3XI+_(|Gt4jX?3tu=JijusW&1URz)jsT$"
        "4{^3^RW7oMg&{!=ED~S<(30Zlzo#1y66B1eN<|lmM0cYfUw_>crS$i+AJ*h6*P;=;CatPiORT!#Qv921!i?e8<U}=%s019nzaW-("
        "eakG*NzZJxJ5jQ|Czk1JkSr*r$4AFUqU)8mWXH^|72EOoi&66UsL`q;{tX-!nj1Cfikt9`G@|7uh`|+TAs+n!k7(bWS<R$rl<v45"
        "b$=PGx`^(Nl3USWVAs6sSWzk2i0i2E+m2T^y%N%X<V{ZA-t~}+j!XNg=hY+XaB*eFD?jp{3H|ntL0TXw@%DQ59$h&;s(C}uVP;D%"
        "D^+t{qB(wNGw$Wqb8{<l@`f0z*T+W$el(Y$19jRE`dNOKwD<y^XZ{d>8v5eJ3-V5&p>LSS0u%V~S^5WenY0fy(NNkYoCrw_gv0CT"
        "N?|1H%}&|E6(lD&{C?B2Zd)^_u_ogMD{pTVy|jat!I$^O3>@x{Mg=}Up!45iG!JyjPODg5MmUxu5VLA5(|TF5QK+mKh^-w+Rv}}6"
        "$KUWR(Mnl@@iQR}(d_p0TB+-nm!&J~$&V*RQM~e<a8I`P7Vx;`kR9%P67sr0AJOJV{#6*EBoQE@23dm(#5u_86G)?L8i#$%nuL3W"
        "Y`Dg{IXKlq^U5lHPbBMj;f;Ra_EzZ|(s1Z0xQ#Hdt9hp^Px=utLm2oZIe1FAWZRb4#TcvqqhnGu+*p7Kc3go0m=q;g&y3Ds?K_>W"
        "9HOwM8J!Xv4Hjq)WHg5=dDtS1!QBlVY;*E6og;xmH&xc*HJZ4bDp_<8O<Rik%1eu8RJf!%B$-~3?=~b+iU;c{9?gW|<e#h^_*<=&"
        "&ge)1L?^|<QuIJK&;kf|3$S2AEzrR+!Wh9+q|Fjx<djeV5QX0df&mAC9xI;FQ<@W6meC@18WubU+t*kfaAAtq)r4LZ;7w<;RT3Q*"
        "1Du!~I(23y0mz$+4CJ7_O0+aHFr^5I;e)d85VSs14gEWGB;pMyz7>)$$sc&cQAms;To{<WFgy10&p&1EZifojIoWDe?D(CJGh1N="
        "{oe<@N($E`mBTAYCrpP^X6eS{<z~+L1-4iTRR)rVaOwp1V+M0S=Uk>tiSui7p$908)q}C6@46CuZS#(`h96?L(Sdv{;>2R{5KJMN"
        "mx27U*cFGUKKF+PY^@h#C||^!5?wg^K98+`H;xu3Z`Ob_WP%^h{`ylQRP!f<+}Jrh$R=Km=j7A7ix`u6MOx7b7>=5Mfus-xG}W}~"
        "7VH}f_yCv@18J8Qmj*q^irG20&i0S@zZ*|?j=Vd7fB?cHw*p!M+gHMiuYg%0@5xuF>2_!tbO>B>{_^KQix+M+IcYTK<eE#aK?nE%"
        "s_aI=`?GUT(u~wY*P&PN9y`MCG-L-eYRRyn<<7|K0~sf;H{|?>^RvIaB-rwa+&t-%-1A%()7k+*VJ^800Dqhpg|{&BR`4v2TOUK|"
        "Hnr%@p|b*(4AIYMW8V`jbf`_t;lBoGDyR_{marGvxC#2*aAH#*cpm>KlM(gU$j447M#TmAgWa$_ft9D%syL6rsi+O8WQ(_nPahzf"
        "F`v$nAe7`MyvYygKN-`gcsQngN9};*5#9y$EJT<SR@Wukh4nlD%dy#NPS}aX%QH7qqrI&%UJuTm<7fIyn3r2-(os>1I!@OfwmS0b"
        "KRy-d#OQ&Pdsim<(etPKYG7oL{==gd4bbL};r4$AfC6;Gf{nW(8QgfnZVw*Esuz1}W#~>0oW1_PF6aL4!tJTyx@f)Q#xNLWVDR!A"
        ")vD!mP3T~_#(5uBY<T6beF)19WCo1m9QTMWSl_qzWd*mE@rx>%Rq>JuBz_e6a!)i?`dh;M^7~+k)Nq4K+5YrG5^Xu`l>e(E<WFg|"
        "uo?(s8N@jY`4F=_T;-FsMKIIB;i#5U2jk0U0S5kyaRRR~Z4J_8=1cH2#U<ajq*grH88yzC-E4$-gCo%W<E&?6jziCSiGwZcLWw>5"
        "f?U8&6$4Tg?t?tK1hs<pa(v5g-MYpa3mt)uRju-!0the2-$pm1Nn=$Hn$`(VJIpJH(?#j#9$0p76T7I_qha^01LHLe!)9pPr-B_z"
        "pu=o=K5i&pw0gNFuq9+uPFIe!rspTi`0F2ULmZV>u*UkH_R6PWLXF{!JEcdh4{7vc;RRV=&489D!FZjXE>F&#!!aq)Brp$b2pfK("
        "P)1*4A^!7*K8s3#V6P;+cN;(h8wza4p_mg;QVjS6q>I56Mi>FnxkpC8)9v9LF$1#4jPKzh%vmyUR^-j0H3{;Cf-%!kYO$F<zZz|a"
        "8tcjLD$Z+Ozk<5p|7KcI2|MnX{tFPcYWEMLdh|A|^M{w+ga^k#Ba!zi4j#{lX$&?`2T#m*N`WIp@|0yQtC?k+Z}S06qv7y*zTR6l"
        "@`Xao-l(wLxRweFP6dr6>pC&3RBIkR^hxIRV*+7>mc4?i+`9?V#_uoxUnlr*#Cfr!`~IbHpzA$wJ(hu)!U(dZbg0oxIBEC2NPYEY"
        ";!{QU#P?ATI{PmVsqKZa|1wc~n5N)=0Z>Z=1QY-O00;m803iUS1~;4O0002n0RR9e0001Ua$_%IX=ZO@VQgn*ZEayMWp-g~bzyXA"
        "Z*DJNUukY>bYEXCaCwc7%Sr<=6o&T$^A3ls6r?X8F03A<(2g>7A&7)bdZq?*<4h`jd=p0t))AUj$bZTA3n5zSiOFk%K?f#aih+~y"
        "Tj&GB-1I>!dr!`(OX#xi%_)8d+LPWT6Fis!d`MtiwB*Pq+RTIybyZhA2M66)uq_2M`cQVKO6=<@b=Xiz<`3blt{@+Le#CaM!sbb?"
        "9&T~Dz8K$Iyu%io8<g|wH7*wo{wR@+Pr1rP#byXzCe>N`CFV`a@JSxpM`BZk96aY`jFie+Dd7rU#+3L&D!%)~1nWVYrahAX;hlCJ"
        "6BpNOU3~&jO9KQH000080000X09?f29bpFm0Q46C04M+e0CRF<FJftCZ)0I>XJu_|VJ~HNVQh6_bZKvHFJ*RNY+qw<WMwXJdDU3^"
        "Z{s!+{yjkchk=SBcHzfn;TApTT+|Qk)_`BIz1ss0g+NQRO+*r1l5*na^1t7ZlJ#_Uvx^?uOAy2sheOUcGv7zbvaBUNaG^j(hl9_k"
        "2Lra{F7e;fARr)JE~o_J^Kd7bM?um95kyh{1d>O>#R-CLsi2`tW$XIt`f9yr5(4f;J_V2C<d7?%kJOD6g<qn8g!Ym64VVrViw5Y="
        "3qO(#-~v4zxMYfplm1AU07c}zcDsl+K9hS~?YtvF!R4W%j|zC`pYF^z!F?ZXTpr*qndG5FN54l5jRh4^7_xA8eYGKy`Yc2*4@~Vr"
        "?J+D8;BE;nM^yv%MBpgy2_GXU;jRHB^g!`@8nzYZK5QayM-^_&vMe3+R`9^gcYPuydS-z^HGPzrsM7)6ok!;f#dt{k)Wasmey6c-"
        "VL)4!+S>nG*vs+Mo~@l(e`>Gvoa?Kfa|@V%Fi3)18JxP;SAaj3PIHo_1i?F)@sL7534#=b<VcuLHkdQagU;BI3bxhtU^YgQY`2nl"
        "wNuP7^_>ur>w;7wez?~OrkGUcfV;WvrZN6|>vN)hgPvJ8@vu1@USsF8BXx1L4eo#M9Dn$<wS1sXZ0R#aUu%1g_tW;S;8N<X=ny~S"
        "+5qN6B5#K{Wmfw4y*-`wJI-`IvM0{4zZ~JHwO8sOyX`h%tsUn(pSC=DY@4cRd)$?6oxygDze|MYbTBEaU+sZC8~)-<?Ec7J4(26i"
        "4-XG54A(@9wH35+&Tu(jt%kU^{jooBmiA(vpRvR4h&INS1mq7MHl@pS3AJ5U$8Rd(8<=DJW#y9VPZqj<<WcVq>nj~UHy?yiu%RNh"
        ")OfUK?f&)E(tc+zF$~)o^jG#|JhN9_@R+MKbd6GVYbc+R3O9V!>@@ASgXVZ`bxWqT*zA&Nqf%eJPN(&zk;`DOlBU>Qy-{Ah^ZM00"
        "FRlLl+t;uD9GB5IFeD)l8IlJw-HstxNYKb*h`1rSm`oz1ZY;!@&5-tnlQD8wj#|tTD}?nj-FS2h&X$XSD90qahGh^{d#n!4pEM^V"
        "^dqQ>L<kmU*aEf%GYL>83<XhNQrKEgv8&eY_Swo-I|vpkCnI!6X_89tM1&kfH}aLt7fHN_#zmIW^_^EZQv~>DCFb8H|Cx`Xq%>Sd"
        "oVd}P--CAqw~7WPJiYgfUKe_wIVm&{r<!`ET(XfA%>t?fb0uzB9(-~Ff=IOctuwK%s4A@ATML8>VMiI&f|T_Rj_)a|<er3@$z>#r"
        "9vehji8I^->PPw<?&&FRx|FILhKOU0D;HxcjWGW3z&xy`9{l6S_`AtZ&FWe3T|8D#D4T4khRhmICLdPoWq&>XXajQ1(q2JrHebWk"
        "?yo*9?NQy7L_XfQBCOSi#bPqHN69QJzu?vacrt4|ci}0^@v~Jg><b{%y@TM2yWGdq31aTCEe_lvc36j|sB)(jsUP5ZLuU|)iwmYs"
        "t&;0X$mg9s-Awg<Zo2<``{OhG2cB-yQxDX%+WmZ6*Mo23L46L&KfnE6Kc5T-aQp#k-_}dZHuKBIBf2tl@C`$*NV{->MO&zafvnZL"
        "85PAIv+u^2^<;YGy!T?aYyYj1#MjTESQSqfCTEggKZknBtXzim=bZO6@Jo<f3JkZBqk+qz<+!AUN4wfh8s<&?QizrCDh_n0W^Nqj"
        "7tJllkABVwys`@8^Lqy-=xK<a2HKM@sg$FzaVo-M`lraf@k#vuh=R?<p>lB{`Kyazj^v`?QzMBUBfF4G#oUaCQ|i>5*)1I`V}`Gs"
        "9}TtQ(%e64&c*fQ-G#HGR=?cpNbD}^0#WY1VJL{Cfkb%Nl7{N(-=B=3hKI=hcnNJ7&Zmosy&lhJFdFyY&*rQ3cv%0Nv2~#*E4+<I"
        ">ZZvqb`d5V$vDT`+2BViT7|)(&a`ei!}<2q(D<e9{p-oRSkqs|<<;U(=U*}})c;>V=*3XKmes34f32qi9$!xIS^ec1{n9P4)Mi_&"
        "Sf3Y35M5O-hcBxGebk*7;NiBQ4oSSu=H!@s<t3sl7Qg)|^sjF<TlT2mTOi=gBT5ztRa-sqW_t5HGpmZcQ%d%x_a#KwbJ~ZSUZ=(*"
        "sk^XpNov7IePM~G6l<&T(#wVTj9nChT3H5~Mg{-@brDapqxFRM`-R&>eE-ib5qN{RD_lTOhcrFPXoCq=kusQw^F2Z5N|psB3hzVy"
        "5ITwL9%`nlm6M3ro97iok}5+1Df-E=lpqYzGQinFaDl~`0M8?3blSX)V~<TUHO>*%qBJ^OEBdXch5bJkxVQ;#-*ifSl)Hnxn}1k>"
        "=I17MAEKcZaJ+7g?D&h;#1-^sBUt}B#{o#Gydr^9oqfzzCV=sY!o0@x5(yOXOWVr(Ra0&O5%;nrfL|?iMN0egR4+r*6lKch7ke#N"
        "LFc=3uR<<{RWnBoczUjXRv$krBguNNmynC(|ALoKOUUOvy(A$o$ViO&e*sWS0|XQR000O8001EXH>CH@8WaEk*GB*VBme*ab8=%Z"
        "Vrgb?V_|G(Wo>O?FJ*RNY;|FDX>V>XXLD^hYc6nkwLEKc+c=WnQ<eX~<9#?Pp%uw*Yoghj$~jxPojh)w*}5u6#iB?^;+Y~feE1QU"
        "|NC_}0HQ%sw&L7+Yd0d%Xf*oW4RBr8{gPyvh@HP*d_4V!vnj+TEW#?uGbhYqrx2+KOR-m0cd2m7C@+$1={u!_63(K?*AC7}3g_be"
        "*P-jWFJHWTk*qg)Q8{6;1h`T(FIHi>N|Sl>{J&+MH4g>A5xyu-QxPEzrK85)<#kqxLIG^T3eo7nFYt(~{&D`#4;LRke{!addoUi2"
        "qm%e(bh3XqJ{9|?^Rv<DG(0{&IG)GibUz%8j!)0_BOv$U^PhjbI}hG}`2G9>rry7N;lQ6;6=9OO6UTGMCyRr_lQ=pWAD)~<N2B;q"
        "Eas;NXGdr8d=4*;_s>qkd3btyERK#24~5tdLlGa0_76|T2ZzT;uJ4>2o*WE(jaifxAwD=c7{v$U{l&33I6NF39i4@dI6OT)7>lFv"
        "==eB{;6En^Comq)MRayHkD~dRIGi6u2WRp45D<?K#%GB5;pK}DpDw=s@#nj*k^`Pv>>^LEg;TABvj#SSfb)4O)=pW8jWfTae}~RF"
        "i25dh;eUreX94`rZ$Kwt949W4EKHrOUW+1$KrPA}Q4E1`17!Nk7qM74;uhEyRe{o%KmvJDzTTt)B{`TVdR+HoSVb#Btf2p+ge8iy"
        "lB>(9q<t`b9Rk!kzYf!1oGfpM5b#1O8zNm3#wkD!OA!{)$}3#L<$c@x+vRBQ?A!30KYKj@eld#V+(J}!kx3N8Ws%n#Z#)>5RgrAG"
        "!4OnC1(|rR4|832;3NyRnn+9Ge9ALnm~aiGEfNt2WnLGN2#Wls^x~udUCxU;u$oPpRGJaub_4Gbcoq8QDMCeut#GBhq=_>*FAl2^"
        "#fwUBkPn<Ujz?h5YsVd;U8U{-=?~*zeg_)osr|wRlobwP<^7{$4>1mgD{&hqOVCVj;QZUvB2Q6F+zJ!e_CLe47UxBg7oNK;*QZx{"
        "$A~#tCnd3H;sPyl&x+N+4<P*#sT-zw7?)m}WMTjcS2*~~Niy2skY?c<AR8&hpR<PkRAe6FvI!;5?M^CzKSHZi3tYwscp(5Z;Aoh9"
        "nTCJfRC}-$0(bHS_^EYWg<yBirk<yuzW;Uc`P1*eNQJp0d5F@P3*durz~)!rWQ}$!FIN{S3at=-)gUD#iPS+a4~%W8L<?ZJzJlK#"
        "IAoDk<@9S&3)H)$tb+WC9@Pw#2C>3-JX=eH0ZNFg<}j(m8id679s_mYFerpf8GU^UYG#30*~B5yW#h%a-7L(W6XhA8t%Z3DoCL}i"
        "?y9Ob<>bwqBD@(c!TZ(ox)cS%1BZs-Dc<B8k%h^dWf8^@1EW=*r+er|-`wk1^B@NUIT+9&s_>h8qUAyD)GSVJ)|TZE9&sfP{`H+I"
        "S&O``rpKd!6PBQe<pu(wXmLWh0X<5F?wtms(Fp!Kwi%1|LPqZV>S&OW15*A8NAQ@1#raK!<}KYpUIb<GU2tJ@N#7KpLdr!E0mj_c"
        ";T4EjK+*x80)r;W8#siKS^a(mLSN;e&m(4kUuv+UFb+46GLSms<nk8`esN8Sg_Y<itpW?oK{isgAR?urWJb7<>p&2y&<MPd@0<>h"
        "iJp9q@d_Djh#NR><($DTUI_t%bq$-Gi{QH`a%7xHma9s0PTYnVyU>NY-?DERSW%v2p26+@H)<2?XyTwd0~#*fztk&thIaQY`;DTj"
        "0)TK?&mmU6eLH;hw_ksy5a-KzG77shCN*RvjN$IhCNC#MxpK}r#$=ni0^uz4G)9L%^Wne7x#WvQDJowh6BN`Tn=~@!9Me1Sb!ITd"
        "iNmF+JhQujZ(b_WZ5tr5l}T1sVFvNsAcaUzVnhJ_R)>H6o=ud<<+L5m0b&fyUj`aG3bspQAr=BWGics)TQ(*CvG--BB_*XJI}(*-"
        "kU=GJ(9G{4DT%=kqGc51E@|QfWi+{$BAkp4;|J#3q}+$$2Ao;!*@WENkEE}ft3eV&K5Ulrdti$^6#(R7=I8bS%wZR+by!>>6w%rX"
        ")~L+HXx0|e3h65VlBL#tz!WlWDW`9Y^tCcl6jJk&l1A4ZbaA}4thaAmQ`<YYG3@r9AIu|L6%A91>3+Ky9BoSzqh}9xg2YYTZwC~;"
        "?;%XgW8n>~=wckbdV^)1HY$a_g^)5F0<$vPEoh-`S&Cu<C2QqfqXj&U+^~ld)VN-6N;wbA9Wt=sO5EwZNfp3(QBGBDFu`<N?^Aih"
        "ezIif5i*bTR3^ic7zqBr)4tFqM|7qqpG;aWpZiiL=>*<GL_4QJ<q6(F$^!30ClfcS{#PrQq|xnh={{~@k=$ZR0K`~f+AP|et95rP"
        "tKotFjEPjB^O=%e^uN9pkQ8gW88AzEG2mz6TKKX~!5Htk%A?67&R1N3VB@)~BnCqX$ddFhJ`6bHm5d<w)z}9r;X!}V@{q0y=uG>7"
        "63EfB67rcKETbfuz7N5L45Z1sJS0=tL}1WB(EF)dSBt$<w`2b6FiRG|a?|OcwE+|4uIM`F$s&)!bSa?rR<#cL+~9i5eT%DwBd#Gj"
        "l9SXaaSs$FQvh-gCNzO+uG9ueJAjwIL(kjFiwBN#TO@Ev@G!y^fCgf*&e5M!x)&Jb1*->e^UUXHr-%qIDJ9}=a>iLNwPMC+HycKo"
        "Pqc$;w1G5%P~b1pwYIADD45L#NM@@%%2QmWrAWI5FBeo2s?wl>g7XTL27pPw+F+m#QJtc@+XhiV@l$RajD^%m3e1HN@m)JgxoZI#"
        "SA+mM=PATefKV!II0j)9)sPW&=TXKegH#DQO+ZZSp+j@-!8U?Rm&FJW>ePxE8cJD?5ud9^NHDHt)69A`6#5Mes?&XKxLrGKccbJ?"
        "6Evuvk`aUXz*mSmN^2-UQBRb2&`~bWt8HBY<^@<c#-k}c`=ek>+w0J+;{#<46}BOykxP5-ns<X*nKni)fWT7I62?Ol*P0oW-77~$"
        "wdrHN$|09-gxOAOt2-4T5K(IhM#e6+o1P8YSBWL4!ks(VLVeq7(&|m$8c4Rm$<@iNlDFHvy#+OfoUHY#tka5Wv2MiM+5g~vxW37J"
        "wmp@zENPNErCLoK<ZQoG{&IGyhi5yMF$iziPF=ia1;#7Yc(if0+mQAs?@4xuf6O4P=B15Obu!J%Szob5q&^;Ta6>bT%Y3@USrf-*"
        "s9)HkfXdBd@B@+s4)B+|3ZTfpgO+HUYE|8hhE1oAttHei3A4La4%8I!ZOymA@`C1~-;mtUjz+4os#-8d3=2RPY_P|R&t;z0vSH98"
        "ZpK`eHtom=wUUlOXn;3G4zY`yAt4}B)YTWIZI<u%UP%bwVev<Y0X%`zl|v^bj}v)lrb+%Gl~PGe<F}Y#s0CthnSmeCCu>$qTc02e"
        "H>ev-yLdq2QClM!e}Ea3f^wGW7TX4jk2RJD7|4h$^UkJ<BLseNIHg4<gdsB+ZfwMkcL^-XEtElKV!+na`<mauR*~2T<Jh|tPWl{;"
        "?4-3Azs0NQz9*NQB}g13T6VOTI=jm98)}BdHF(S@gbd11%q@n{UYez5l5hz?=Qt=Wbwuo05wC28JWM)#-Zaac@5!cRN|<y;m8=i9"
        "F%A)J<(%4vK)Dd{<kQT2wzsCIDUT8v?+udv&K|5>)zl2u4MxU211eIEOJyN675(Aoe<&vlKepmuicF~5NuUTV?46SdbHWO9TML`9"
        "b`PxEqBXF+-PXr+%S>%L>eQcWt=aJY>C@{itns=uxw^Bef>0L%wyIWEg6)E+yb2r{rtL19GMyUjHC=kuScKe6&1ezaWp+m+@(pZ1"
        "q;#L6HMp#0wF=T?om6erkYYx=yX4AYO_@2br_R{SQ}B{??LlR%JGCU0@=lZ#;0@g|X&63HtZFqXd9&jg)h1N96(-;^uBkQ<8`g}U"
        "G>p-2tiC{lx(5yt%&o`~)WtqRRC)5r0#N*;J}vz+<*v>gq-AUh&}qn0Am=SNBBEZGDn^mHRH@^SSD1a=25KOv@+*;*)BVxmsddao"
        "huQjG2_SYLwY&SJ>EJmuFZqtaA}j<IvG6uW&#Ug16m^#2Z4!3QhJIk|yq)$?fN#eK<I&0f4$)H^qDC>VW6*V0nr%%sb$)5q_?b*|"
        "_snId91^yJp2uth1*tmEg<X27mp^zkq{E5DdK4n}Dw2u{BDe4qpzeNCPN9!PX1)`~aey=?P?#%4qR`qsqy_|-8g=EqBjtKh4hpzP"
        "^BYdEQc#xi=nk1W8U3@A*QCmN1L+8;spg*A%1~)&E95<i7yGp8Ft%Ze_DmVzrz}Ak9y-AEE{IlP1rZCP51xvewJz_Q<|p1%Ks8gh"
        "6{wAgggkI}qa`<Ox5H2*%yG-9@@RvL*lG@1NK8__NNyWN(7Vuzp!V0R0<u*#u&iF80|S%4E=MyoN{x1&mvXk9Wp2|31mEP~q3&AU"
        "H6TwPVKDzF*`bRo)h2^=Q?+ICmx^4OF3N|-JOaGAqZ#zd*NwK@AQLy-gR+qx|H?)w<OFK>8s>0xb__7g4gXzHE-z!M8%B~u<&U#m"
        "9>s%usj-eg4kJ=Xd_+WE$~a&`!ux7Nx{;-JDKAW@8%flYY))$$K5WC2WGZhN8r9RNWLsG6Ro`vG?imVTh^$9Nz<+lsn{X28TXVll"
        "Yd{m=0xSai>8HpzY#^zB+7j}>_+oMqRwno;+9uTYqeYExxpaQhkeObmbf0j`Z4?|x>qP!mGW%E_(4(Z?SFBlLerkI{?`BE2wLCX5"
        "PQU#|Zl?9JWn1=R4{WbJZ5!}=cLnUj{ph_7wL4-wpt4!s2ZDbu_r^0y6~?|Z9v%D6(WqU?NSp{e0bnzBmk6-gzOx;Z8F%aV&JmAk"
        "OH8C*(IICGiWO#)#@{M=0cB8uT_u}-7v4tI<mXsZb_Y$JF=2dU@A;`sk9&rM(E~qL^6QpGvZO1@DLNYm3@N(W<VnW;UszW;cmnW@"
        "H*~40e4}rvQ6EFP$0!nI*>Z)_LE$w4s2W1LcBgz6n0QI|F_gEGZ>zkBR{CAn2|~PHB+JI_LAGZEv~3QVNI3)>pd)YW2*~Mer$hS>"
        "@M7Jy2-=245apr@)Wx_m02dN8PT2%@6@b)kF{54SV<bp!i1|Zx_kiwlwhoIL-^~D<O9^zu!7=~bbp_M$pl<@2+X^xW`uv!S9iS0m"
        ">J~;kFwvbXn+rrjn&-hb3H^A8e0De@lG<_Or=w9X*4=}VmbUGJowrcOsVj6;&=HJwv(zSdRzb%ySG%U;y-(5rhBWxsMMSA38QWHW"
        "?2tu{qykN+k7d91FlpCow}RKXpvDnIU68rFSFW-mR0lkY*XO|GB(Qe4hIC)vKHTC;7?UJ}EJBUbRRfwIW}sJNlrv&rc5S7|x^&TV"
        "86Y~Vf5M~nCK}sT?MRIL_%N#Da9D^ED|-<_YAM!vaTj2WYfksLpObfscD`z_!Y(CABl(K%hjWy4Gu*xHyEB0I!&i`Kq`V;I2Pb_s"
        "jm?i9&2MF1(v`M4r#<fC@#L&W*Lx;BX;`~vbzLLk`CQweFo|jo@!K~-w%ir5QAjDsdmuk>fosr<Ws%n9s(p5BRK;$cl8oqv16%gq"
        "ZNRD1sDAr)8&isQ@G$TlT9RufL`RHY_9Zcs+P(Yy@yqY$U(f&3>XA}s(pr!WVvIyP;#{0;<1U~0WCrURKOy@d>VbpUs)tSDM-|ew"
        "v`&L>Q$^R(G!4B?{jhgK-JQGu#U<P6xv}K3(>rrlO<ktoIhZvEeN>pD^Ttm$=z1*7V!jM_tejW_=XG(Q%PX#*M4`cK4BAOX;JQ#*"
        "8jtLR3;ZI3JP&zqpK=4rYC-yroM@>Or2F_mOwT1ZqKJ+G3QtlO1S%yDf=*QeYh)$f9RdiLqyRnpeP`BkQ`>vTz#5C|1d1(GLTlgc"
        "J7!jz+kZn$TG@{G!^~$fN!H}cboPNMpZ?Z%_?v3-e4!C97G+c<8`Ps|5X5-|%uy+CJ;xI|0vl3?uAa-M5@%IYm%hKuV*DmV{8cBQ"
        "klaP63>t92L_#2Ed#)OBZ9oFiEhx|3+mlz7_L{qCuugTi5mPKR9$m77Nt(j*=gXU;YC7dG(JBWams9zj59WW1uuiM#=rK%8Z};}-"
        ">p#F5k_}I>^o1Z$0Wt7$CDP5*eNP_<ZsZ5p{L?^b7t&5ty{g;KT6lQLp@;SK-sM1@&FaeMX?AU$-&{g?CW>9<^;G?Q{y$&N|M>9n"
        "{L|Oq({CToFLYg{6Q3Ri=^@P{{rnSb+KPfjE~8Oxntu?diYj|77ybzXv&WD<HCtuT&z#+)k9#r%K7SXxYSWW!kl~xV0d4kl^0}jA"
        "wqwEG{Zm*Vy6!<rvZoT2Cy2pBz&LUG2eI+5y(w2vo}5nhwRg793JHvSRx3O1`{<c~+n;2xgq8JO?5|yVC;GhJ(I34?Ncwz|7ligz"
        "Ui@R>)E>aUp0CcxJvYWJGvVIrlI?7unXJDvFeGk+{Lx%j|2^2R7zcE#C+(AL<J*`ByNy$g!8*p=(y!}nJ?zq%lkv%F%F!~`MXMd!"
        "bFB;R%$o7fzSQ^Z`@MQajq6`=y>~%33Yseiz2#=-QZ%nb`!DwGdOg)l3G{~}ox?Gjuj{_uZko0y{!AUMG4vEZGx~oLFr`mF2|v+@"
        "X;1%uNY{;ywu|#Cs6XU25+8oxwGzT>eGx&IQS`N*hk;7}VI-i!BM8Xgg8++xAkYRcOM{m${tr+~0|XQR000O8001EX`~+qRt^oi5"
        "`T_s|CIA2cb8=%ZVrgb?V_|G(Wo>O?FJ*RNY;|FDX>V>XXkl`0Wpi^baCvP~OKTf35WXAyABH_O;BM0E9tvAxlR!v9oVG_ZmS$~X"
        "J<N=hK7M^iyLFt>=pYzL^Swq*(=4w(b$})p^uTmP$FO6xF7Cj^Ogx~W7S=>ylab5ixBl*H|9ty!-v-+@O*5a(X9K4IN6dTgw&21s"
        "afU}NU+=bVNCy3mLcrFfm{}R+^mO@xzF>AKK2be*uSpCJL-O|2vm(re=|1HrLi=mUoVe1>XU3xltiLb6P&E4+<VYgjd<MEK4L}nc"
        "eVC5w67s_Wj!f2>DnmBaOgDO1fW?eLdBgPLL~lEg%nRTWH_nd5gmeKPKmX^YJF{q5t5QY7butIyiV%rASoODeo9BnkW4~T6pzfy%"
        "FskD?Fe*ziV>4B&(pIY{<m*Y{r2<00=mru^Hx$V_Qj^$Lhh#EY(1!;ZVYfA&fPBzYVhS!|PTZbBdu?Ke-u+goi8Y@3A1s&@j>t|@"
        "RIsflZ~`%9K#*dLW<3;t3Rz=l{|Vq+Fgl@k$M|M2W0(HWW&zFL(Du_nfqs%Hm0&c(B*kad!J#U~=|a=0%6>SVZ2N}$Vm|u=P)h>@"
        "6aWAK2mk;8Apqv;?GB_1003|$001li0047xV=rQ9W^ZF*Y-eR{ZDB8Ec42IFVRUJ4ZZB?ObZK^FUvPPJXm4&VaCx;F?{nL>^?UB}"
        "KcMV~lB>)(Tf5t-RXW#}Yu#eWXQ|zCZjJ|&ki|1avH)mX_42>(Jpd#@3RIGAQ!Ww-JiPbte&fNO=gp~19~imX$xV@yTb9S0E#>z_"
        "#05|DyAfeeWtPUNByo{2cw=!T(;^>{b^62-lCZqkra6@b7vaknFJGJxQQeljh?x+t9$5ZBQbF=U5=sQD<*D2eLG$FccnZmi2`LD#"
        "a#G~kjy!HyPB{Cmf?<{bI?wYEXSyv54j6=No(s`Da@IVyq?;khw&gm_T9~Rzlb0{nyx5Y2N*ZTW2qs8FBS~X9YNrfnN#!O>Z)>nC"
        "cty_C+-<rpVw&Bt?UsgdkxNcv*<ie8Wg$|K&u*F1Qfvz8T88x0TJ|LCl`+kWJOz+Qn3d6nip}71;4$7<TPe$I7pcKW77Y$msp2rI"
        "@^qcDBuditdIU;Yvxoz;xI}I4<%_>JV(@|H?<}9Jc*TO3FVq;(#<Qa0{+fv@ljD~!2z=69NSeniN|P}WlIu|zla;UrG-t>qFfk^#"
        "MUm-=b(P0xW3cfO$ml8ZZ!#}(rXe_^qR164G9hRrnnF@k(lN?7w@2|`!q&>hSyU?1*K+dR8yFc(MuKC9P7O%O8QM>3m@}gi>^TxK"
        "PfKYJ|EXT2IncgEZ!=~Cf&Z}>yrH!Lv7vwd_OCV!&9dS#LI;f+2Q^ZgkaL;|Rx^@r8HgwXPb~5TB)iTEDgo#($9-~!sRzbIy&Izf"
        "0LLF#5RjucgGR9KQzPF`$g!DKN~Eb^<QJM%Y|44TeXm}ytw3J4pc)i1Aoqd>=YUW6^;<x`m<-G85OMYZFBC=BMICKZu|?ykxsZHk"
        "A<|)RGURRkVYl;Jg9(CK12HQl@mt<TWU(|4ZGa0`T3FMkG6fs~L@vS*$lfHA70>rH<XiSVBHq=>>hvNyKbgIqo_Ql<FZB<o-B)OX"
        "oAI~D$M0KBsF?0DypPjwA$r~h+$>iotJ!I^n7@4o%wV_~X>%iWu4*WeBo*i`N$9m0y9`>M7s6K3(hojcKC}7x^+`0nzFu7Kaa7cq"
        "6A2MRs5X=@$xlU==r}hbn>2xBsQ?9m1V~~~8_Q)1f+jfwM<ehC-*Usq7DSqXWXej08rfqiH%&4NR}5o3<-0RbevE+#EeULK;+ec~"
        "g!X*oDZ9w&mf71b3iNr*VLna;A!nk<9+)3kc3p6Sq<{%+;~~C@aaVVkDgc$QP`>Kr5MQy1Pl1z!r6&XmQvqu;r3`Q(g)`{U`>5+8"
        "qKJu@mQB>Kc?E7bNIBfzCn@*!i<oGgvZoY`rMRzi+}_$pC|+1}y8{*Kt*Lws*q~|xefeG`*GGTz0&A^61?LRou=nHad~tg6Hk$qd"
        "U+1&=Y<V#~i#+nRDoH~8`o<5!4STZID~p2FM!>!o8U03P(*<{;fFUm|ceV_^B>!NnROYpYiY+5p=z+aeIi%BggPug_8ZD6Kgi#I!"
        "8Hhxp0cdl?O@=l;h-4D&Io#4T_Yrua!bX5K-y?^R2%%9FWKGiyipf`Cv{?OBAVvX^CCxH5W#R0`Rz1Qn<3R0)x)aQ|T-w^4iJHq@"
        "F6omp3s*oq<n;T7_Q_#jmV++RgfuE4b+Sj*v>~)h!g{R*8KFB0*zU7e4|yM2FcfnGBAD#^YmOy4s0)SlNV9t%h1Yi>@VHdeTk~<r"
        "NG+JDs?p_eUGBoFgsPA^<fX7fhzhAdNCF(GSM{<sZ^$1K4fQ^dn+9eAUefG)AOC*nIIxj(sF4E|XB9QD;414;aBQ`e@c((6NXWg#"
        "c2F1DKO1&0(Z_Iqb3nV|G#5Z?%XnX7SlK>+U}SU(>Dd=7gW?`paj6z(!#kEUp2m?{r34u`oIym$GY0zsk#uztVim6yav~w~-d56u"
        "7C()Wa7jod7V5(ys#mz^h3J6B4Ca{GK82uK6N3A~?z*P#&<|?cAKt1W+QY9gx0dA+?z)z*Lr3^g*zVcxvET_h|INL!wEM=)nmIsq"
        "A5wmYb+Csa*?SK$=*Ep!2?N+Rt9@8E(TheziDiPbr-j<B^mT9_5bAsPunD(_CoaleltJ>sOg&&lHAl*Jhe7az!kyGY_tRw3=0-RM"
        "$M=y+G_aFcJ}J1j1x@@Q7#&6nG7ys+6zcnrK-My!+#G^DL{9}R_q0RGMwGb74imD81&K+NRjyMdK11z>Cm68UyGHiH?*x?~)Sj=h"
        "g{|HV<vAQR_I=)~`z<^1Yu`uL9*l;%pvwj0(mx)NXH0^sp!<S<2=>-^7_?m4gQ~W;<8&L0_t$8VVD1J64P2-P+`!-(1|G^2b&>@a"
        "(j%cN%de{(BhN8Wxm4D3k5nwH7!O_RD)Wa}QS7MFbh#4{>!;vG{%glxEmaz>_hF#GDN3w+?y}+*yw2apv-a>H<j2Lfgj&>pw#)!z"
        "JR)CxuTLCDO;hhkZSu`mFo#RO3PwcJaUm~x3LgJl7qEM3vx{7P?9t^hHn4nSI8ak#Welb_scU1qe7sEabPIJT%O6r+<Xa&19mt(r"
        "oWJ@hl^50RtCOo)=a_`aHQz8Kl#r`;tBb`PPOleM0jW-;$`PLNV0+==k@DSgHNCuEELQmT&)NKJh9#VLy0~0kO;0_22ELg7OU=JM"
        "i{8$Dygs>p7byf*{e|63Kn#9A*5OeHVYV~T5~Fb;LbIO`Gd^feg=Dsaf}&&`U-O#FE^2qhak03ZP8#p&1&g1T)9b72#W@_^2Vqv1"
        "SHpADLz9<RAgIA<E&lled05U?(~0-mbAnu)EH9SR>Dk0Pb_2{7)A{M*Y&QR?Hj|Ol&OnYs?Wr!>8Vb)%EEI5-i8ns?eoemqHH0t!"
        "hq3w%e(**^824+<2B`@iu9N8}A1xHl&puw%9GISiVKw}Csl0!z!jD;xXNk70#H*pye4vKd>tG(pdEU{iZwB4z_~5k3JY_~D2D|Tc"
        "oT^{Lt`;HN4<Q>$+2Dv1_&+ck(-Kt7qN0*zC3WM#4T*8^;Jj|%O<o^60D`_f>z3djYy;Mx|F+#V&<m_TG?o>`PDgE*x*uQn-S!-+"
        "2GeY*>m`$S9R^q<Z=1W}9WN@ti;MIuP@-H4?3DR?P}mRYcZ8hXr9bS$Ziuct-7p8t#B|zSQ)|z}0dQHgCQ>)5$8Dr|?#lrR10%|3"
        "{re$wab*c=KXx=l*GI#~&3<6e{;zDwG;v&>7(XTxOgfIKVF&#*=RD+?{c^<xf({@i9%m49V&(#=UAuSx(SzJ?UJZKnCw8#9e!QAp"
        "P8UC~8Xq<d{kn^;8rFuxj$;^ROsx6bF53Sfmf)SA$=#~_w&H^-&hd2J^i|Wm2Gh~Ts!G*;BQ{>IPL@j#BaXq$KoG>k3lA%E!nAj("
        "nnd$Lo)=Y~sQr6g3cxbP8&CI`1|WK(Y8kP`(V?umyI;6l@_oVY-<s8T0g=@QM~rZgqWK+5Z0&yjm5ggM{Yr7)n{cQva$lXfV6c`*"
        "Pb3;jHyL{9yuP1xlpY#q^lP&K)BAo6rwg%R$r~@YIr`@K_;}n8G##ZmU;^u^xZ1)AD*c^BxxSy!YlUxq$0SRIyy=Vey}BXA3%>SN"
        "G-J8{teSzAlskPxgY$#B%S3Q95FEU}Oe*|C4LCs9W?+#JwHZ)}7Ng5!8@rxuybrc)ro!HZJXXL;zw74wY_V<dsLS<_<H2;V%;FF("
        "W3<oh_Pu5k-niY6r901}rC9+${a{1s-NKZ97(jrN5eLyWfaH}WMo?<|qN0=8`@|Yow%yC-9^?hV_=7g7KS;%icwJ>#6A!vVB5&##"
        "y3b`-H1(fn>}sx*D#Om-ZD_sODzlmH1Ubj3lXo2*m;i#$uHaMi(MA#Tw8u)Xr-An}8WLax#BY}P&kdgb>JQfP4>o^h9@x9le=TIp"
        "J8;`@9Z}%0{~u6G0|XQR000O8001EXwiE8C*8u<k76bqQDgXcgb8=%ZVrgb?V_|G(Wo>O?FJ*RNY;|FDX>V>XaB^j2X=8M0Z*Frg"
        "aCv=F&2HN;489xiI}keASYix24IZ-7QeeZ-73;7ay26kpI%O+M1|=^Ay1x4?J8hgSEl(o({ZJBx5bL+^N^ng>ill+dU+ZO!R^7mt"
        "%WvmjVM~nAIBsS502cos16gHyjj(Y>gXBPsRKXWQh%`wPYCFdPZoRW<(y(hmIb%=-a@H5JR<J{S4#=`LxI=X^+!(5#x)(zl<;Gr3"
        "jZ@NWux(|boDCcupP=8dlf)N{nnssMOB0O^Y?%U9ev9Lcs_q%$v)Q?P9_8v->UuHGRe}!;f5)M6Sbl_aXR%BZ7=qo_!wRmhr_>|b"
        "A@vxte5wqO2Ya{S^<eHjVPa4-YZPRCkXB(uI#QzS3_BW2@(4er>2b~MIBUeGzB8m`K$y`>csQ;5DDu5xG$^Fj*=g(y882MOu0yM{"
        "!-?Xb7rJjdf7FQa@P8?Wn3;(3g%?=7z-^sIQ(4(!<+zGJbttj#zpgMa)e&_ponSqlsG;>eV<o*JTAfMbkL_l!w>t*O8w|su#Ffh8"
        "<Ohwxp3#P)z0<_ml=#&T?r{NlPd-%cZcGnZ)4m7151IIFMRDuMX7Rsrcmrc8ZrQl8I7|sb<b$47PLtiNL?4>v`?wVU08mQ<1QY-O"
        "00;m803iSr<!oF)0000(0000c0001Ua$_%IX=ZO@VQgn*ZEayMX>Mj^a%FB~WiMY}X>MtBUtcb8c~eqS@(j^UEKe*-RVXORPbp0<"
        "$;{7F$jnPiElSNxPE{yM%}vbAD^^GZ36`X$7bSufB<18MXX_~`De-av08mQ<1QY-O00;m803iS>kw?nl2LJ%%8vp<(0001Ua$_%I"
        "X=ZO@VQgn*ZEayMX>Mj^a%FB~WiMxCZe?;|bZKvHE^vA6S>2D@HWGhdVE+eJPg1G~cF`7B=M>yaS`-HahqTw1Z6MGRWfLorK9bt("
        "_S*lwGyIS!S#~zXU4aA6d03kt^F6<rVUi^IezPU|D3u`x#aQ!16l<AcqGsg41XI*-DP9pFiDCUoD%z31|3qq99~qH~wDg`4T}#Du"
        "ktE6O&FxLAWKSAuXx&k*nI;_XnP~|R`86X|eH<_Cc`Iw$9a!JfqL!lNhj97L<B2KWGhu!>vHHS!qiAhLe(%_ca`ons_NR{J{^vgw"
        "Yj}-1t<gXc3nTBD;J;YK1Fcr7o|+2GA{(YE<i#FMPC7}O>H#EcIcq9^lc_b@x~6R_yC#tQmQgb(_EwsAj5beil#&W9aeMQ((NZap"
        "`Gtw{M>VkQ_QtM}@2sD`8=bSey&>?i%q!mL9WeuFe8&X{SQN#3x37&nRf<s!H{geIt3e}s8?M47x6v@Sae{c@kMn)}*Ra+{Oy!QI"
        "?p_aHyL?W3O(=iFlfQD}h5y^r$4amdu_S?w)T;zYNz!djwwteW^3^7b=oo2eR|OR#ig4x2%{O1e(Tu!&eaX}i=x?lQL}H#$(LmPJ"
        "vOf)mL23*dlEf1y5Vl}#c6vqhzzyAZ41<Y4#=sRL7&!FAJ869SAbH*rZ_IPO$zREK+5<vDn#hQ=t&WLw+RDi1OIoil8pF}ab`w2C"
        "xn|@i+6~NxYMM;&Q0<52z#u&yYp@q<lFZ$Efau>39aT?YR~s0GXXCTGuqP@FIWwx>bCISmqR`37i%5$MEU-cCK=kY9UYC&oS2Y>)"
        "7H?LKtiY&M%S~lvt|lr$F%;9P<Jl9g$y+I0D8x~Jjq9Fj)^WkAnhInH@xb^4tL7;gST^i|L#&lV0Pv)a`-v*rGsBclv#f9@MgiT="
        "!aSYeR!I^#NR#I}RIz^z95B5C)B|**9ayKSJ}`bbno%UM_5+uLwq_bV3?R$}7l#NA3id#|)ZdbWfjoiZcOIA=Cc#1i0edjVs^dL3"
        "J{Qc)bsxAnx|&gN;mefep2nf-p$90sqXx_f`rSp2d7qO8EmOJ^ofL=MTdZ_{vNL*M3w=Kl1#fGR^q!@#HK(?hvRho&2JEOu8#zfr"
        "8F3T>hio;vFwJd*u@1b3-{uT1MdZRP3suhPXmGqO{w3+SHZgw84<d+F0GSX+;=s~5vMjw}C2aJWM*_D`c83kEb$3`K_=r33<B$jl"
        "0CG`dl*_4{ByOX^JgYi+;1KZn488p73Qe~GG<`|5vY?!~kATR{ce|Xt+`eBiRBOGBVy8aVbqgtFSn3%Uw1ae7Gctb86I&qIB7rJO"
        "aj)8n7S@NL1k^*%v7qXJc>8jj6+os3=I7U3Vi)vuVxmbck(E=Roudz-6AJeQJw{wGV2Wxl2hkt~Pb2`=zS_8qNPt$dPG=8<?$5VE"
        "?5zn5)&dj9$P-~(I>WszfEYA$;;dmvZA+3m(`32VDYvCLf}<>77VB*4vXpIHp%H@y!{DZG4m32v5Kp1Mor;)k&Y>FS!~=}&_sFjf"
        "KwQ@6w)D?SO^K}!bUss;^Oiwxi$X*08i)a?@h58P<16qUf_P7#px;qY-znG=%aLIn{1_UD5^Rc$;)wl0kg3P<eLxNvQeNt!Q`E2+"
        "Pc9T34KH99n3Wfr^we9(;;LU8>$Qy?zj)TOV*fZ5d@GFKjRM23JVTN3<hml~%w6pGHGe$~`sRc-O@)M^ItK^ZIU%A4VlN}jH+_q~"
        "f1V!a2ojEeyywG@3sp4VoNn^YWr-9RBxDfQPNd;&OWL7pc?bQ_ffyJqxqy%z?4U*l9acke3*q#$NY2V&<l;~?7p2_d#T8M^k;25s"
        "AE98IBaq=*5A3;8V4LkDuCNz<SOL@Bf6GE3!Jl9W<_xi9h6;zE^gSCg4K+|9zIIPl4Wuf3Lg)Y^BCYTKohaF{WJi)gGX<qKv2qIo"
        "OU_Kcg;CHF_sLU!OF{Z*1JfZ2Pj}U3L>?y?PGA+dRosIiXGEs17Bn8yFKFS@&yD>nsb?u=6I@Jl84fG%+ui0p`D`-G9?D;s+MhjX"
        "!JU)rcs<S{zVgflA;&HcoT7={QDP3pqe8M<ys(7P+C}ni@tE?liF};dv1*pK-b9ZS6ZFMfS8CP7ZsoaR8rj34Yt#K-w%>daru*fK"
        "&~855jYR_dw~%|KyLMh%d(>H``1hMFM)?l|<^^ON77a)^0S<m(o;A*qp^i+<-LqQ}C*NWoMOR{oS(iRDq}-5;H>#fN9s?u9mK{tM"
        "!30LU)RBZ>iW6=YF)JYu?f3{N!|U4o4P?=)oG;DOI7B)&25awBV|dQcAr{^1ed#5poc~;$TZ_T+(#BkJSNuo$i}7J^^fjKq2!X$M"
        "FhXD}9Y1tMj(4>1+-{sTZ0bk&9M3BHo}bo{Z3Z^W%t^<J2SthEP|l%2v0aa#Jv5t^XSRYv4W|bowc&MTo3*I5J%3M(O5Pka*3sqa"
        "17f+|eDUQ)WGSBkDP_Qt-G8{Hb#E7ce(QOu<s4tTKdIyVgPmp+5O&6>Bq-H(^22I_8Mld_ljLXdPaub_ibixr@LV=)$@PBhj3D$n"
        "@MSh4{J+2XUkBGn%oUuCu+oS0${`EDYnN95IWhuv+!vvJ4zizy&v`^P`mdhbo8JLYO9KQH000080000X0CnPo8z}(*0OSGy03!eZ"
        "0CRF<FJftCZ)0I>XJu_|VJ~fOWMyn~FJE72ZfSI1UoLQYeUVLW!ypia_m+GI#5&3ix=6Cfs#2A9S5*kZ#HIoU4dbT1eX&zJ{ZY&Y"
        "Y2M>w-i(xT@6KqnKSBsvXfmB+Fx->&BB1R(<c%;r5k&TbV8{$sTpV{I50N`yXG%3@q`Z|<)>U1d34>^^+;-${3kiJQHsovz!i9vy"
        ";;0MiK$!E63xOTR{bX1nm~yF>bm~+hd@r3M&Ai?7%W+hmW?_W`p#f`MZ1enDr{o89L9%KQRYLE~u)+yIEm)=J05k0bXDA~x@HiQk"
        "fGen|*?la-X?o|ay4DoCIH<|C@}96SzV_BZ!UINzPd*g|Xd|ZEj>2BE>BdzlrM*{5Jc!r266MG{AIr_Res^n&<6ic~urlh!^KY3y"
        "@PC5cl;8Qr`deLn0Z>Z=1QY-O00;m803iT4l9bqU2LJ#*7XSbu0001Ua$_%IX=ZO@VQgn*ZEayMZEs{{Y;!MSa%p5|WiD`etyo)c"
        "+cp$_50L*rWG_-HE0?u<@ZbTxWi7gOU7BG7ib9|z%I2(FM9E3|u;0G#kdjQ%a@KA$g4QArzjJsl-#PNh$;mHOBg=wkOo@uuyb;Na"
        "H;pK0mDM5@wJ2i2(z0fJnQ>ZH0V~ReHEY2Vk@7Zc7_Z~CY((6&wFr-oj*qTsQ`T6SeYm`44X;<CVXYF1<pSRmw$qZ<WzJ+F8_6^I"
        "S*u{45&$S;b^7e;#p^fMuU=gLcJ(UE6JNu>>7{|UVhe!rV(p!UXHWfrsTe!ON*fs0rBb6CDVemm?(&|IcxX(OMb$RDO{rL0#A{x_"
        "(C>kR=eQJUDq{&8+9n5ZR*5>s0$&MMrmVtwQEx?Ze3Xc|OhnC=V$HV_zQZ5<sMl6CPY6bZ>`Fk{k0Q@GENBFFw2>XY%&Sb~kQ52b"
        "JUKa`n`5!88<z8CeSD-n*W6ch+is(erzP8zh|c&aEDFY<rPx~zQ{22u2}oQL&hb&4ai!SR&mZ2Eb?!mx=Q3$C;m?nc7(TEKdoa4r"
        "cOgWu6LPb5-?Em;N_^k>Z7XV-N^G0EXZ#+1sYR9XSYJ5oE=$-*F&0IX9#Mp>$kG6bmCwmx0TY#)votGtgCsZ?v+sI~=%ZT1$MeJC"
        "v~TSJD8y+T!@th@t&poVA>yL%Q53mo0AxH=7D9P2!1rwfQ69IvPCEPU=AApQMWM>tXR{wh3zm&rUQ12|wqSQ*Ro;3RewYhhc(e0>"
        "ZG@<xYH?k+!m_;<D6@jyu}2+ht#0bZ8=xK!xAP%l-N=sRk5uX}s5+kY#qnInXfnE5nrZRbAkWMJltxjY&?yO@OH=eFe_nMN&kLO8"
        "t-{hvQy5|Iyl3@FS?=iSCejgpEt5nP6m58o^r6OdO&h4b#yk<O!k|l~C+yXe=fn{`Vpnw!BUW`;m6Wc1jk=f@Bg~@BqqX1(a_?Q6"
        "FCh<{9mnV`OPrpK{+EMdWTsvZ*_)KTLsA7jmnsC+_5(XT-6^zC6SBE&=4=;_Zi%k5<V%^!=H5EcF>JI#ETb*Y=B#5;M&!w8nWa2N"
        "N`(lM;dN9lu_W^>f@Oc{wX?G)ey~H=#bXgKho_Srqf!d+EGutme_%i=;u4%td=_=taRgcar%56Ey7ajwCP_+R=yBDeG(nEg#Zq|("
        "Z3pOh#KJrF-C={NqMJ=b^fhO{uLbO{DGI!*viqL(7##eozQ<btqTs1NH8WV7v~d#-r83D|V}c0|Bs@{tfy^CmSqX>@qQ38fseJ33"
        "!AD4af|yRUi4^J?6mbn6N*GJ-X>h=Q%d*TMLY&o!X#OMBU&GCI1uSegxg1{Da@Lb0H@J*wz5njW^p1nb@KvVPcawN@6(%wz4wk8d"
        "E5VPwg^n?g#+sdjR8Z6!&cicys(XH@seW|E^_&$A9JQYM%%smZ?ES9H9DtjMO3<{wpJX9;?O7KMoI3?<DY;r)ayY~_WwVcDMd)jg"
        "w*MOv5`G`cJwCec2+MZiuWx7Xv|opcN%ul9cPP?0xC>4{P#ub=<;X8QL23D*O5FKPLj9t-_a4;b2s&Oww=1H)-)P~N=E3V09aqr+"
        "!K!|in(xMa(6egeHF^e^MK>9u8y}c1{n&7(9o_kwi=Lv2vcmJuG5KKWl;PV9-~2I^MXNztS5j84a0Zq2cu$o9!Pabl%gra<Hz!BM"
        "Mh|=5N2B}`vI?WRSeQ*srQcpmkCFe_E>r`pYCL96w2m(n%emTE&BB&;y2eAJEPCIiu^)lPgh{WA?(3In@u8g?&ICAhOQVW<x?1x}"
        "FmK75c&)26%0w|LSKpZ#Tvr^zf}Fv!Xyp*r*k8KiVPvQ20#gstWh)qo>?=9ff1N6)`sf1$#~z>EVi*y+ot~Z}8xtARMeyQ~$poR;"
        "2t6k4u&(smf$LH;z_(SR|L;fV-YH+_Rz2kO4@*lvvO2Ya4mI*lI<g!cS^at}y_3J`JErML-Wc8T;HDQeEyo)JTY}qOaC<<|rkN{Y"
        "JcC)Vch1ndsps67n@sR98GaLPrMUHM4~@HjFmC>S*kGVl^d{mVO)i}4Z3ei(h5r_goUr}xaJH+S=~?+-bVMt}wCmkCbzwzT?{8ei"
        "1G%UeL%B7~=pJPjfnit+%%8WJcY~K}>pt|dJp&i_<CvbiY&E!sUHtYW=v>E<4B<HsZEC?T=KJ;j4f`-fLYig2lru(lEk=(Y=Rx;w"
        "Y5tvm`UhL7Kas7*SHx=*MvvWUJ%t1!k|>o~=KW<O2j<PeJDJHMJ@K7eOw23~>OsS{!R<I{Df43o9l8uvnKn7U^Q`0g_oKz^+&_41"
        "Z^H3x0Vn@{*Q<Jr!Q0OSO$^09K@%F+j8LQewDkpSdw`afJl@#cG-w(o*=be%SGS_WxI`tb+Ok!6HJc`{Dd_Hv9icb)qdD8_wM|Tf"
        ")(-^zk4R-PXP<%b2cm3^gO6DMTUFux5wO>mc0HNtpJxXz>Vv~{2IeZA&S6r0|2MU<6il$dAC2#MeDpU^O9KQH000080000X010WM"
        "3cmpW07C-+03ZMW0CRF<FJftCZ)0I>XJu_|VJ~fOWMyn~FJob2Xk{*Nd5u#`Z`&{oz6Z#EAbe_p#@T5HABGJ>3k*Yd$bwxBMWSON"
        "vM7*DI`7wyj+><`I$*j879Zb(bX8T4@7_01n8ML(qzvBq9_0=~6ph&{bW8(ik3cCvOz9}fPU8tdFm<c1KL9Ora7tWPRkd6!7w3}T"
        ")cNE@P#_M4Sg%i0(({{@`TkGQE_UQy+T$>|TD<GJGwPiBfKHE$-$*y;!g$Y2JXJ0iA76AnUvQs21>{E@h{tBR06U(t;!kb6nD@wq"
        "W7Y;t%^bzKqBZ&cP{@KU<0*=n@**qX|77(s1kA0dQ<P@bSZ^`Qh44&rgM}nobo@NI-drF$*rT?@?ab+BGWzbH^lxGH;Z42Mv57n!"
        "C)Xh64%r&oAafcUIN)e4PslbNkt41pyCOa5TIql}{z6_qVchpB>rdaeP$)$|))V_n^9`<MhUfjjtkH0uX@ct%88AeZ8n*0WHs14k"
        "f%8nJpNcKc1axgFibcTXU@URfQ!(Brm!b`L+D>Nxnbfe{E@6P(XwXs4pzOTXe<{Cy7`)({3v@I60o5H;Uxt!!FcGg1jmqTzdZuja"
        "PEXlP>tm^_+vVanP)h>@6aWAK2mk;8Appoma~4$u005;6001rk0047xV=rQ9W^ZF*Y-eR{ZDB8MZ)9a`b1!3PWn*h_Z)t9HUvzR|"
        "ZgXa3axQRrjaJ`}<1`R{uhjoxFfR$$4e-EAM0tS&Apu(HwD;0oMQ$=_O%pq^-R><*|955_r%l=(dwz*B9?#EjK3fR!!CO^&@c7B!"
        "8Qr{qlQnwO&R7TDO6_X2P#X(=i_lplnbKH+wB=TLEdAcH+^jL1&*rm_vQd@v=s-3N?C`~X>$b9^|22{o1gT%3G~G)Ee;Lr|pN-u?"
        "Ny-kCCMU{4YpQ+2=7bPzQ?;G39xTphwKXkt(r+8JhA6+M-;tU(6j{_9Clj|>RG1DmADPc8tWRkcT!o_CV!7*#(mt8bfYw@;yS34{"
        "2#)jMZvo#d!0l95Q2dq4VQowUOZZhbjukD|joNH|!8M)k&Sxpy{R9=2J?|Z`w)#Z-f-=V@Yoy-n<pzuXSM*w5+6{Wr8sf>Er$QHO"
        "C`o%A{M1oLs5?t|?IpaZ3ppHQv&XZHJ4*6JFAt>;_A}~f>jrzgR1MmANY2?(4=X+lO9~m=HCBZ;J1TSbjGVNHr;q*TS^4d`?1}R>"
        ";X<YhsI6>~0k9OE7YnHHph{edvfGOkq@$M03T=K4MjoVU<hnu1(4kLyg^Y2Qq@$o)<($$R(ptspz6%k@v^n2Xtti2X4rfdICs$9C"
        "T13mEn0V1pIE0T}%lFnAo7CdZ7{VD(5TTgEYqYY;e|S2I^#9s2ggG?k!f_ZbN;4LMI#{@dikOU3N_#o0jyoGO^82pAB<!Z})iVBv"
        "0J2_s;qzv~*D4_&@gW3A`L@!R+mPTZg>=lpnY}F$UW1S0VgUH-wg02IwtkJ!>Cgp`64^vU$`vho8T?<@Hb@vjZ?#qYu9a8BasMEp"
        ")p@0*S4M*))}e_VNN(KoNG{p9&1$1bt>Tv0Fs-W*$av7Yeuhs?yg6n^&U`eMNbC}(;sQ{8p^J$76Rz_sBybha%FHj;J09PNrWCHM"
        "#b9qBq%@k!Gv*v#fi^}qRt-_YU_F1j3h824zt~TYHlB;D-ZxG7y@63H8u^0OJ!Z*c^)7wP95!_!z1Jk!6?Cut{qLdw><a2xOMm<!"
        "&Dewmhd4ESF=D+3iiOZkmThvrP@;D-GGVWn`;B-*9@k>VI$u~}d-eTO3O@oDwl|;NKG1KjCg+~c1>lj{9U?@FpnQ?$jb==Nok_xm"
        "t`er8sV+nHOIW2A54x3M9QbmM44U^8BHrSF3lHHU3eST~n_^>Sbw#JVP8>WQ@VA+GSdrEyl-!3s<>cF^zlL9wB{%T%{li^LsfV}1"
        "C=h--(F_~K_b^R6!Jk`6)Cs@cKfwJT?_-g&+YUyyGPYGDeZVmudf@G?9={$$cop@bi#%SHi^+bt)e5qvup*W3hAVZ%#(|p{l0y~K"
        "TP6<i^~hE}t30(FLplq_s;EnPLfM`li^x@U;!K)j?J0N}{4p**L2pN_lBvohEKNMbl^Sx1s(g0eCJfHcaO&tya^mK}GCiUG15ir?"
        "1QY-O00;m803iVRC&n9a0{{Sn2><{u0001Ua$_%IX=ZO@VQgn*ZEayMZEs{{Y;!MTZ)9a(XL4b1XkT!0Z)#;@bZ>GlaCwzgZEw>s"
        "5dJ=C|KU_5l!!ERFeX$~Bp5J|U?A8xgd#V&wuZzGwgc<G=ls&d2?a#HB)0FKdwh4#Wj32V`*Z_$;ZjxzpN#5C(<y`xN`A%Ca4F!O"
        "thyRuBNe=rVmrT-!r-m(9+$Fd74A@IGR;p<PEUfT2Hmx7&9MR-Kv-k5ZxP<S(7;3mdM^ySg$i4<11^lDUBG6IRmH^?bjwO4z#D})"
        "*W4IXKv5NQq%!M=qC!#b0jn+Q%z<8E(=dyMTg4oEMwb!N0B_nF8>EwpP-nB5<+@R_v8-%BGLXu^`_{U#+8&23BUQN@uW~`i8icTB"
        "rzd62w6^!WQmoyD3`ms-k^AhW%hMB}hvNPtKmU~+5Bf}Pb-=fe=$gX96%K9**B$vOkQ=DDHe8fu8Zq&Ch%)^{mTcYCOzqQrKyKJ="
        ";V3QP9lPaC*Lb2-lG3>c!bI0`ja=QjsCcuCTZ`7dbDn@kn_4(~^zg#9+RF#efnE<DT`c_i{CQm02)a_Tm7Q6_3;zMFl54i+H8=ab"
        "Px~QR6>gv?xZtKJk^w|xy~+NZLy8P5OC~#-4S#%bo*lh#a9*50&W^C&)RNirMZO5fDa@b3wG<c?m6`=rlIA`2bj&>;GbX(8#>Bek"
        "PKN#W#HPOi+4SFsG}`A5Trw(%RJ>&MMCE<2PH7$(70$x)6#2`8aWW>I4RO<P;MW`#BOSWP{vv}q5?#s`?+diuzJ(fpU_DKj^Eu9;"
        "&&Qv`zvTj%vW<4=<We?ks*Kg9m#$N><pCRHcBD>AV4ZzvkCyboqPQ8XIn%3GtkyWaLe2K5C|jxtFm`=S%MntmS7$iC`0f5dE6y_r"
        "CkdNrtQFGUPo}EBNF|bqpN%)9X{q8~x7noQ!)*N=+kGJpw%Oz6kIy3!kc{pBrZd#wI{d0a=k?XsZdxd8C5<wv_EZ_$lkyvZ<#81!"
        "Q*(M1AKXsO@vn<E^p4RYSu;~ok7)c})F|RPdF;+$2EfpmzHk2pI>3HJy04g$4&#*iI7-PNh9T_H2yu_+qf8HrLjzNxj%hd<(l(L_"
        "--ILp*X|cQ8EN*RA7|6r@ZNA0D(|)vznQ@zn+~lxoTX!+h-3eG?>DI@<9H*wZJ)>0AkgFhmj1<X5Mt7wMfS>-Z6v#q-7io}0|XQR"
        "000O8001EX>gOZ-$N~TWB?kZiB>(^bb8=%ZVrgb?V_|G(Wo>O?FKusRWo&aVWo=?*WMpY>XLBxad6iUAZ__Xkey`O3aO8y+wb{gD"
        "(GX}<0%=32>v%z^a_#G4aO`k)R<d8uIZ4x{Yd6uCRKD|f-}l}5k|eo4e>($7&iko!dEqSD09!{2LZ$89JNR}r2OmV`!DAsv1We%u"
        "5tNhu?Bn9{^OxoPa(TO$r@2a#BpD8dgAKVHln6o^;XQiLj7(vF6qDMs;7DevwIDoL8w10^hc0o%P!DJ?mQ><+IH=EHp080UX7|k*"
        "Evp6k(uCP?0Q?IY)e5b23TdUV{?1_SoN3}-+&@Z)f^rN<eg@ixHYF3pIo_2damM1QVWn^bq9{zYivFx{$6{5~G)B$s7J9~F(vIk)"
        "iFJHV$5tj^WhhjgbEPc`Y7^c}I)%pta(oQa_s`nH4O(i1PB*yqFO<Q<8aIv*q_eEIl!329$Pio+bgmze(r7`1_|(({?dt?#W#%Y_"
        "WyS}}akH%pkz_qNj$-c(=5efKhVpme*k}+&3*X=odY%Chw6iH(3S$^MFMWWGGsbOU{<lEV5#Xjk`I_gj%9%T9Id#@l(72y=`kFs_"
        "bLeQgMSYirl-pY%yt>%Md*`i9H7}t(!gpaxoD(@3B|GI;*AM|4mKhHe{J3{gtV<)Pn(T04t5wK;LXDdy<CDDBY(;c4`u3F{;IkYS"
        "xIyCfW1qzuA;L^IguPnFhGh@W32pFy%{gI^CLX3o>w}y7r{qHq{1pcFHu2mj$h1;u8>2ebEwNF**a1(4_gWq!+64M{c?b`$hw7-z"
        "`h0(R{G#HUdRIWV3w+a#lGEbj@chc4<r)Ro_&ufdK7xxwg0{@jjK)tpv>TK@aX`Zr&-btgmkx!ttBNnpo>Dn~W<X;3KGcgA-P*X5"
        "+h@uHAyLOVf;YY)9^k57xVuVcyZ6&Asm`$DFZzT!!cdYuz70o5-o>ObU;@dmj1IBzKTt~p1QY-O00;m803iUh1b)gA0{{R52mk;n"
        "0001Ua$_%IX=ZO@VQgn*ZEayMZEs{{Y;!MZZfSIBVQgu7VRUJ4ZZ2?nrBzFB+b|5i2grZmd`f}T>0!HM=rCYNhoWndbw#g+BGHL9"
        "Su!Z4U55SmQGO>&x&rgbAgPb9k7SZ0w-;~cAU7y~lup|KCAlZG!b${T73ABnB?K#ienNOgz1{=}B&^So(j-a7qwy#wSAY^hNF%&Q"
        "54tEFg<cem8d-28H>tHCJXqVkUg_LPVb)j_B9+eOdflQdewIkOKpVc5D395DDe~MI)xNkw5u?6y;U0zBEr=XNPvg<YE~E*={6c&A"
        "KwCT;k182lTOG78`WL6`?9vE3ACG`P>|&UM57Y>)bP6e3Ikn5Y+mgge3%}vcS=={zcBK42txAI@Z4}lv1AVRV8$qZl4;eRZ<LCtk"
        "<i4Yf*$n_h2%H(qvLI-UZI1f|`>RH%J~^(p<)^+W6Z7F}k1&N(Tf>AX&DO^v8ZGsZV^iQ>`)z>=Ry)XrWP0r2_TpVjtwHQ_=1Y28"
        "0UoQ%Kz%yp;|P5`^CJ|f3!yM)PhOH<VW#(tCZ2gYKj7&ZoPT(MPj=W#xP%CiHqw-<h9=X@=X_5AXgr=|M2pf$JCvr!epeJ2NXr9n"
        "pm`8os9>v+-wVZapW)2`aun96gxqa=B+p5$ujm<{T2bJ%m+j|2P#mPx79eQM+BdC9HBog*PQty?&wmFHi#GsCxS*LRBi+TyVA5+2"
        "JIclH`&mUC;U<a2=ggoW%Y)q$IEn8Eove%#t>W~CU2--(sCqxxkqnLYtS5BGV`L``DRlHFyRm)+vtQZ*^M2$_+}1N!hI4J(Gh&=j"
        "+ODVVviLf?nLSn%WMDoI1y_*uX}brOgdT|vKgX@l-1Ay(8Pz%}X7_Hnm@U67W;c)LaUt@<xLY_U8GH&4&B<!}3)c_b>^H^bvt@&W"
        "utCVR6$T3AWyrzs!P0?k;$Bml_|NR#^UOd%!IzVwpC=^G7-<c9&rT?&VSpQHc7sAIC<XB_1b$GDTeZC*l4-Dr(yo-vpb5>bK6cn9"
        "@q{N+Nbc$?4sLmSr(q-N8DAk<;81n0QKbnp;SW$t0|XQR000O8001EX#-x%R#|8iZixvO?D*ylhb8=%ZVrgb?V_|G(Wo>O?FKusR"
        "Wo&aVZDn$2Wn^DrbaZ8IbZKvHE^v9xSnY4)G#LLrY5&7R5h#H)+fIPcphy!8IPNyxZDU`isd5u{HBIc)cDm~c@wfBXP8vH+uXhq3"
        "s5GsKpV!~>>W)UEtJAY{;B`qQ%SZwE|B2>fqBZ532@aCZsiZuk32>o-z;CoH2^3^QB_37<$!LkR>G9F=(I0|w4NA;)N!|hBIj9AZ"
        "G>2Cw_fSdk7tM@N1^5zJSrwL*=4m7KM!dO9;LYpH<oJja&DIoN-H(k7#-C0yk}asRV)0A1tgJRJWJwD4o>+5BQdLNcmX(BYxH|pj"
        "<m&WW{QC|KyrmR!k*Sl5o1cGwd3X8q-Jds?X_>=ZNYD!k^c~5xF*L8Op`Z!Ep{>e-l}uYD3Pp`ZBjYgoAfyH@WVQ&OQqF-W;M|ZM"
        "ALVoovzit8^e<!ebj=i-v4ZJMbbJK(L5M|8<#dg{<mZiyJIa-iiKEErAqqu{`Sh9bTs)tH!RBwc7M#Mw{zIY|#y<s&=Z+D*c||ua"
        "EDw#Iq&4aBhgLOYq*kOjaVluOC4r|x%t%pe!1`6t)Z$}Avw7%R%5zr2XA^h<2zuaCchs0=K9Vp+;hq#Vy_8bOXmoFA5jc#sS}G`O"
        "rC~uZRUT$UXNv@3Sk)$DA5HdEXiL1u6np$|`e<~9mb51yG~}{z-egCMkTn4mdyk2lvpLp|uigs~Fl7tM2FO;5clP51PCuyn7Kyzg"
        "ZP+XnGYXYZ%oNX7bwT-)Ucj)IXOE7^ahd9zG&w>f;ZHPj+zB$TSw<(J4EM6p(`7Zv*vfmb*$3y3#;@+}d19S^JcN!1geU*YK?Oez"
        "$cM2n*{hT$SJBi??t^3bvWL`6h$4zxkw%r1lS!eJCn(IXU9ObAt1#Hn_$+}I*6~?`o}{{#9HP!&7lw@JDDo{%LcKADhLsbP)5Fw!"
        "4tf%B#I9Rq1yKra+SY$%hr&ghZV~6{6}Cv!-1U!JY{Er1EnrrNY>A19Y4_XP3)ev@ksOBsQJb=)8V3$6Se{evjfS*otghkBVLD};"
        ">FLx1#UPP%k1gZ~fo*!;QS14@5l+16f|6XhMmd4+dxy()cz))cn}e;;9`C^(Y+?3tTGbVmQJii)jk_}}ijwijb20IqN;)@^!02Ra"
        "V%+Z~!~J9%Ta!sR8EhNa+fEwOw#wm)5VAws-quuZz_GSUa@JPSY=cAJody~WvZ)7!S8n*!qw`Sm+TJc*=fkA)XMm>lts!e4dz3Be"
        "NY#GrQTRP>vfGz-{Ol?N$)gZhEW?+%2^%gEad_m7I&01z2+W5T7#$S#nsKbp59Hd65jFXhAi1ZkAD@-A*r2qPIGnKX)fS>XqOH8;"
        "N66d8=Ho=J|9g?5*1JN7T|EqGgQMANoxGc?+iQG?Iw7=DlQTO0ZWpaj<oX=`#HIsCYj@zaDN^hnwrz6(8f%vmQ<MrX2e`81M!dk$"
        "ylwMoKfNs(cFXkC@8T%?naF&96K_#bxM5Q@xZK)Y<)1{he^d5#?x)(fE}KuYYod@T4)O%1UYl{|x%DdOn7slH-3iuFD?X(5*5!a<"
        "d}cZNZEuHpP@B&@*iGYu-Tc<bbE`J)yu$8nrCo!`z*OlE5nbagFdEp)V`Eo%8d!e*;xRUEIg-%Gfi>6js1rPhrxAS6>7h%AbK2Y^"
        "ai&)V!&S!$ZZ^Fg*PGli-TCl@{-wEKw3i#FG80^zdoIytyTTV{)zD%|d7#XY(RUgq`-V#oVviNKy5{BfT449l5W1vQ(d?pUo`luz"
        "<nl&EPf%};rE5l?qkgT0lRF7#Z(joTFs^Wgp%L!fFNR0gl5SiiJa>DyBTKf&6E~!yj`u@eG`f=A-QquRWWC42_5a4gRhxuv!bPh%"
        "#Mhf+(X#K^I-M5+voFSx>T*%4mDJt)RIO?vDSjy_;r5!fq_i?Ac_(Y^DPMz~fN`n?a^-qc`J?wqin>y*72N<oIoZV`G<%s!T2Z2L"
        "iPuzOEv{+gr0o`jab^7p^cXesYYTGi`2{KHo=MpKlAXQNUm7wYy~4%@R@OzdYSBdSw!n6=HK2}P#3lya%pZ{HzNDr>(tJ8+MG^he"
        "xRQVqI~p@SAH~5!a<LXHhih`(ZVhm8{R_^+Y-u0nc9Ll(Y2lhL0#gyZMdq-w*9s|~Rl3*3=LIe<QDa6*_+~dpl7=T~Sl&~-Y)=ce"
        "q>(%D?ZvX15gP{PJKg*Qio)YP0G#egK|3>Z8|@${fuaP~!z{hpZ%_7Drmb7kyV+AMYSq6-d`$ms^I)JWcwd_zH-dFJYVAY1e*sWS"
        "0|XQR000O8001EX9+RSUu>t@9g9ZQqAOHXWb8=%ZVrgb?V_|G(Wo>O?FKusRWo&aVZEs{{Y%XwlomJaz(?Ae?kCcBH`6-c%sSp<_"
        "2p%e0iqsTD5j-Gdt-WJs#Cq4=UAHyj<C*m(b<(r~EXDC<E@#e}8AnmHK6<f&+8Wv9o{WY<<lfj8oH9OvliONRP9C)IbdBIiw?I{`"
        "utM!c=RpqEN3Wjt7D?SEQ54N*^BJ2N>w%-?o87xaYY+~!4rYZl6=YPHT&NPON+h|_1(h&(Fa)#|(p}b=KVa`I!*-*~+DKF%8(%j*"
        "hD*G9#=m3X1C0CLE*C{%lpLK;QTWE<h4G(J$o9lqV*}3lELXxgIPRU_2j?!)LhFPt8imW%d<J|-EFeuulTXuyLsi7!82f6kmeD*M"
        "uHiR}WwLc!PTp~ZBm{srU7UbrOoKXS3~8VMZ~c&rG*_OIbgO~yFQ}cdF$nKDH;a~5!u{NRyM!licOFd8v7)C<U}fxsGtLX&kK=Rx"
        "kKgg>$7|3A%D|%+wnPucf^4;@C?7yt$VI-vzCFgsR*=JkoG(IaMlnR+i1Yf*Hg`snpLE?S)EGO7mQz|JcC;u{l6Q+;D&xD`hlskf"
        "=mXgUd`*l745{OZz`a=T8!n8-1J?IL?%MvHAdT)=l~5tmE=RzfID8#{Wo1a`wjWs4YXBK4qf5u(d3Im^pCp4EL*8v`#qjC;1ca7w"
        "zJ?9^?nzS^dn2qIS~Uw}x!M`huI|YQA1ZtGXfuQWLW<fmuCs8n#|=N(V*AI+PaCD6oBeGxC0ah;kR@G)W#~Myjm4{mk&V^QXQB42"
        "Atlq=BJV5GDATH2AP1L2KzVQ13|!pFKdkG78wbK}n*?X683%XOhNMId(%fiegao`L$IEhqb~22Q)IQ;Vee^uA@L~J{bUI%%?4lhN"
        "3HytLqF?>jAww3NyCk6JrSGuH6n`Ry7N5`l08mQ<1QY-O00;m803iTvMhxiy4*&q9GXMY~0001Ua$_%IX=ZO@VQgn*ZEayMZEs{{"
        "Y;!MeVRUJBWiD`er5oFB+&1<-K>mT9eUSoLwcWH$U8vY?H(e~6v_TW}#V`bo#F<g8TSUqBwCH~Oo^wb_Bt?zgb^vEaB6)c3H!_>e"
        "=ErO(?>PGpuj`DxWJB2;S=kJH-@`!LT(LfDineC=yxboK5iy?adC#(8;0;cQSktnB*InCZ70a{yz~kr7o<I9=;B4%~(DST@@rQwl"
        "A<J)Bd*2B5?#=6n^<0$2m{kDt?VEQDwqZZF?_S63r=~ixg6C~vZuI<FDPT=kWjQZm_CBlm*{<#Hv%X;BkoCM^Kl}oy=hawn!M5#i"
        "V1idW0-X1)fOkL(;3ojV&1QpGmWcPjeiaCR*8`pMO;v8=nh#l#4Viky?szXs&pe=`9s3;iR1<e)#f6$S9J{jF>ybPYgR$+<_W8kn"
        "ikpUIf;D=Z0Z<k|W(}KbC?u<wAAqOdwau>FN5-4Cz_iM<Hm-3%#Cw`4p8&B`K5NLq^Jfe`Gq`_EVs=OVE#>njJMvx)W?feu)4m<D"
        "{+M>HNO`?gLm)Sv4e9M2Jcq~uLU3E|+B~ZWhd3u%G$yYfI^LHxZ2rFEIcNwX{J#8+uPg;x^I_r;&HiZ|x^alud;akf{=UtI{1CAZ"
        "?JaKrPVqDEva)~ukkwt~g^IEV&{dUzWHe;C>O1w;w@fh_n7-#lI;pP)upP^)NPlg?#?re|l-sf@hocV=kW@AxqqyDv%5${RiM|eL"
        "NlB?w0N=OW5t;h@S;2R#A&X8WKLY`@Ud`-_*o%ma{0c0jtysc7^dr3G4T*Dl`#snt53i*3z*xQ?tBMsU5VQfoz}^_B7DFQeqZ1us"
        "2voLUyI->%2&3(fEUWIbqhO-|)N#|8%nm`8J8e(F#?;eI$UaHf=JWXCtN47Q_9qR>O2OIBBQUe(ulv631MRyEcg4U*++|e>vIBpR"
        "?SX@0unZwdB<y$-Dy<cQ4?r<lZLqOKKx2SYjPe4eAA;RRT1@&oGB&Q-d)^1(A2E6m4G|C+f#8JP3QWlg5fCJB6<Lv@GYp8|Alzg#"
        "4!g5gn=riI;D8d_=4N6O32WxVeOZ8GOW0?ZIEYBF1Q_=E^4kpqz2ofVt9i1-Bpy5DM=-yqF(7lBB7o%L@>R4pn1UUq6<T=;EEosR"
        "<U?5$ypf;?J!c%{1H3OvKtj%X$zCwuOaszl-<R%O<Rxfi;-a&!+HF>))A+>~#so`5nlYfCDMhY?)ILImO}AxH_EM_>+GNMcCabCq"
        "@?M#Y<ob_o!xe1h$dZMhj?u;erF;nF{?>vR_h`}*-4Xgdsm~x<3b-5XM~AU*)Cy_xc?G(~q$#{5r!zV2z~jgbn|=q8i}zqa(qn*E"
        "=^?0stl&ir3XFjhLMlXTu$!-O(;T=3lR3+FIE}(9#cUN$5iq(S6_*W-!>NWCw4i}XE{yjOHA=)1LkBO=K+Mre2x*;(TVp`Re+Vxg"
        "h!AqqTYgL>%M%ooG5o^#tpu7{$hiyjB=R!uph{Z&gD+m(0#hb}Lc^#*n;bmRku~i=dUa*%xAcU+;56G0Qtl?>&de%Sv|NG`V<+<r"
        "J6Y3SOX1;V0%C)hF!1;m1Iz)7cs=kK?)ES#=3AG7o_`#{6b0iCT~+4gFqxjh4i%Mlg-FU!HAK9z0UI=KWgbxRW)J>{rC?$ftuZK5"
        "$1UCJz{j${S$K94kygjr(HMK=EhM#~MDz7%^gQq5$RSE563)l8fe;m_-(BDS8d!*h@m<O91HvZ^mq2{!S3rZ&gb=v4M!-7`4i<gm"
        "Wf)U%gy}p0f-Y@kAY4j8h9ymmJhCHH(Ar3PR3ybQ#cP5g9XTY!{KhaPaGVY(43nUuwC0X@U>d?=Tg|##gLSumf^{<(FY}17zrs5F"
        "Obn5<%n05+ir{I$GVPnxa%n+QQJ3$%rY+U|`r^j3fY4RZX%5Hog(=svc({q1v7Y@GMPIQ7Qd>;4YNRurr={#ANioyllPhyM^q7SH"
        "B5;YZtmYtK21<=IHLh^Oh)H$wtB(PPUPUJIPk^PZSO6fN?jVg%r3V8Ob&T&((?xZgjosJ;QX)u%3mEs8H-Qy2B0E$lH!9G7Lq&RD"
        "Hbr}H0GIrnDoBX|EwNQ@-?w8YjD?zo^HRmZrz{qE+fx!R1yE;0JyzPC+G4~=a0bkWq*hV_^#f4ifhB*$7@~mDP;pej{}k^cwu4Gx"
        "2sUSBv)hF0nA{(D&jYn8&YC0bf6IA?qQ`0uv_P(j)RNygs7zXww>*#oCU{WwZAM}$O&mqL8|v(VfEOtJM8YC0KSSr&N^Zn7dIF&J"
        "yh8i=iS<nr@5V+Y;&B1RZaWT++q2e0tV<*_NgQ+5sMcb#avq-pn{<ht&_w;Ts21=8JQ*vj69XH+e=1{eO_7=s<$i2O5ro(yLRYcO"
        "A*>xJw<lCu>4D^?fT)aqZo2;g5<aTz1K<(-7%Q`6*ID7zdxIRFQp7EG*LE2|2BmZAdaU1}G=mazO19c=0G)JG{1nQva!6C+F(_Qe"
        "^Q%SI#@A~w^Ik|vv3!mvV5ZuWDFGPNN%(R`bbl~$n_q#`|A3`>axAC=f+!{UPv~x^^U6h7eZn4Ut-Ht6T9KydczN{B#$tPLDlEMU"
        "sT%<%H*~Nz>S#ZOq+Y6JlQ?_mh@qLWUWver)Ivk^;1n9CQSqcRpxgf~(1Ib3gsMc*+a`()#44EhrfhiD2h)s5wHes9%!I7&9SJy("
        "*wcal#2>&&G4R0y>^DMHg^dD>*EBA|t0l&AWkT)VV5Ma$Cxy$!bKR4yRP`Q-M-u5%;HVKLBfIQD5CSm*qs~KY3}FNrdiW)kc7?)u"
        "Ehf-&pWc#kRZ}h~{kRg3l8H}q=f=~D%xF;xt~b4J`$7bv<&@_4IL~r51~Rcc%y5l|DztRK2C-06o@5nNe@^ZyMjW(m>qLiea?>r-"
        "Vmy042|a^$Dq)m6-Zgl>l8?RM$*R+p3(TqN`!u`tkP)v;JX|$X@nL#lsaAv)U%77*tKTf4t9Yx?UBF#JlwT7}`SquWk-KVg{)u~("
        "dCiAITNs?yHN6Nho3XJbF!UZUZuWq9KGtHaA(0B%8M}bzDyzG+E}P)oQeLSF7*nW2Ecr&gPbjb;sj<6XPW#B!P2H4P^R!QF&uo;>"
        "I-|T`n+ZHM;_QG&RIQF|ZVPF`d8#eZQJ=ee8R^am6^|Q!55;%~&VT8K+?~#EAeHnxmep8f;4bGl!OYoq+A^6%<Mc2rIVG!2OZ-gT"
        "Ap|JZb5<@#;LcLFX>BA45Sqkbm96vOKS1uR%cv8j`)3>*^z4UEe~HxjDUMO93WJmk-XGeklD#!3^OTCl{@it`ZK~H1I?2C9UW_SX"
        "5MIY|EYsZ^SryFoAqARN?Y<nudTZGy7VU-|3201DcX~;RC)&ouY|_~jt;7<CX{ECJ3)=PjAAcp;GJ3OiQ`VD!I1?WaBqy!R;_Bb9"
        "G-Aj)vdCvZ)sz*XMb&E%crozg(-t^S-<sgc?%5}a!zl&&>hi|J)TMKLt%OEf)uR~T7Z{Dxp0?bhr3e(qp4g{LY)ntI_vGP&Nv>zI"
        "eNwSvACADl_xp5kN`o9G|EVJkf1Js)1|dLN!m*S+!B^XLD|-H}>psDjlaAY5_jI?7ALopS1aQimgFqbj<8XT7i~MnYg-wSmqX{qF"
        "sb-u?khQ}$PBgM^rPkb)cs@#CQfwN*%+6|}0FJnNT^lA8WQS?UnnR%8eTqx5u?Am@Xd~x%LPUcH=_*bv(kM%)-Vm_#{GJrgV8t>d"
        "S=x4bOMACy5lt{lw(9?xT7AkYNu+pvW!-jxg!3EoF3jU8<1*$D*eRxaQoB!<sU5LtVdCFvtDesDGN`RF=FEl-Bk-FqT&C2=h^ZJM"
        "#bmcFJf}&%d~V9GR|tz7ik&3P-aw&qYQ>tRol+eMPixHrH9g>DmKKv%dYGiThDInmi+neVrZKr|m2H8=VK`qjlD&S_3d3GkN_U?E"
        "UoVhU(dk+$X(_DSmL@_mHtWYmZzbv$JK5fjW5MsroL^x@h<7LQZrsq;7;4pix}o2U;I{<lMR4(@zKJmw+2n^Ls6c*&F?PA>strz>"
        "Z!zJMp?2!H23}&%mk(*jv)i=hb=x1w;)Mn%*AU01a}z%=sn>3;QaX&vbRAqq>;nHjgTI$y#OTH6;f>o|lufd0Z0-c1iU1^9fMQ>3"
        "3qnI7uWjA*-=7OM;@Aua4%PD6#U`2%q%4gPI{!RkU*gsmui)t`eDg)<2FJ&NUz)%#OyIAG^z$WPMe34^EtvpNFIi2zFfN7lgPH+_"
        "yb#NCXk#$mX8A4oVzqv{N`oS*N-+eqM?Aj|bn3eAv!Vq3C1G|kK2jI5!`7n#2k(f;Q3Df2+Ya03c+{0aVMT?M)WcH9kzgOjI248i"
        "dXJ!|eH9)}+hC?m9hX`H*~i%yPcQIA$i9Jedwn^ehe$d5PCP<SqHw@G5vMAo)sx!(X*d4J1K^=gG3HQ@mW{k~qC703sIi+OiSjZ|"
        "%FQSDk&vn2#V12`P-w}<v`Tp!dSlg=s8Wgi@|2I%>`bKi*ZbqhKWon1{rA+dCd~Ef%Y%&UStk{50vjXM%w9p`+=Igcl+zbCD`b+3"
        "JS|Vi8Z5n#xrDi3R{I2XufxMyj;iA^40GM&CsMS^#I#UQHBe$>?xZ5ARK!Gnd9g^b36d%mP4&t|z3?qbw@1t;ZR}rw1#@}3j|B8M"
        "L3%3#+};4U_knZF)5H#A5yqwz4_@&yU!edSVXMv_biA5oX}_rho>w_%=;~hKY2bR&lVD@N*0@2XZX+d5-eha;>ct^!*W23f>%PNs"
        "Tdz+Z4}?Aok;RloRzh=*AfB>|$zyP1cuEaZ)PKlGu`dqL-D+|Rv>ocKzfJiA&&R=;vX;iJXK>SLu4>#Xzeg#S|BAqT6#X4k#X~>l"
        "1Cw`Zp~lCXiCh3Yy65^eQf6I71>5TIg1EpO7jWF|@P9E(_s3!SUD=Lv)wqXTOPy+`gp3ZwLw{W4XJRqgy~IFWPhTGB?_&FsSM#8G"
        "3e#XxPeDhM2MhlXP)h>@6aWAK2mk;8Apomc;kJJV002!F001Qb0047xV=rQ9W^ZF*Y-eR{ZDB8MZ)9a`b1!gmWpr|3X>Mg?E^v9h"
        "SX+<WG!%ZX)c;`2OH!t3yNFU1g0z&bRH>8-K_ygGk&`$xW|KJB&U6RFf9D)u5@#kmTO`D;v^x{~+|PF|p_IP78Cp>ZLwaV!o|9_F"
        "tLs6E-jG(ZTJ&4O`kJts4Th_SvTqMW^bJ?MuQ*X$8>xU&_8F!0{=NJ6L^nuf$d2h<E6VBn?^^cv?=?zxWMBr5h(CD?Un4@`SMB^w"
        "%^GfaucgwNeI95~wW1L%M?!nG6pgG{yX9TSvQE~#)!A%IVHE2%XdM`Y?}gzvrkF3D;pQFW8I+@F>%u)N?2R>25F>i<Vb<|ZszWg-"
        "SxzjRWkq=~T%RQ#B!uV?$ST=4Vmm5cxaUDS{?@#4zG;!oLcQevJpw;^;MK-`ta$|oDze@mKxmH+@;9>Q{`A2_z84kGX*Jf2hG*t5"
        "@Z5$fD;v<nJbCJmjOY*f6>D|iv*WB*Qg+2wl=-vlv45;Nuk$Bg!?*Ogdjg4dHMT8rae$orV-$*$7r*{+`DjqQX~lMDz9GFNwfxwl"
        "`I<;YUcCL4K$L56Ys1>MWYx9H(RJHNZZk1CJxOo#N+3BsNr`CI2!sYT4MA@bu!Y-Cjy(|I(W&@MXq)OuidOkBhEQf+hz042A}G;H"
        "15#vvdShnq?F??mAILyZ@(PaMNOL8}zJ95cR0;iFnq7!QO>DLigc8wmR8?F?H0eeSf^UR28BG_)o>*|+WhaWgW9`U;MojZofTdO("
        "k|<trL@etA-$<=kl{gd0x8(6zIdAyZ6@ox@Y!oJX!~vc<JUF~0?q8ZAQpjgHarg=GaaZvX#)a*fXjzHDi8!w7Wg)Usy?v0h#fwGB"
        "M~;Be1V90CGrXQG@cPz4w<VC*bu61FPYXQ`SetnLsfwa@7>`jwBQelZCN(Ek<ohd-iCRzkmB3|C4tb{qII*?O6&n1rnMzd04U{Nf"
        "o$O!F@U}LPOp-DW$jw^0Bvbh*urcPpsFBaf5NCJf6{a;;Ye<Y!eZGFHic3Gw3VX(z<@u?&qYL_448=#OYMoofjo?Kf3Z~oRBf1qA"
        "L^r%QT*|pt2eFGgU*zh6DF&h9N>9ZI!aAFF4rYzwd_xM7$LR@Ox7c!1z`!B{@D`blRDk$}dS4Lq?xH3<AQulFeEIm!++H5z4}I0^"
        "v1wq!f;4yH-u-;}BWVyzu8ssa4(+I?qGEyS0cwY<CF_vB*3^#mBQ$$#oC*H%hZYn64k+&58c;em3|LKa@)z~lK%d{9n^w#@=%j!D"
        "$1w2<zNemme!x`tH0(HS|3}=!5y}drus0I_TyZn1-W_kqUo#!$Hv?CqgIogdTX8)gW^;6-g6dhgMy$g<Ba2#8CQ0MV-f3{he~J%%"
        "UKX}g*Kp?u+SExXL=EZ)b13$pS|jY$c+OkB3jySCkG{5(1I`~B6y33bKf2}zvS+QRZRW#F$6dT(b_TSJgEoq@4xF?X8ZQK%8I1*E"
        "USo!0dPk4nzJ6JN7`3kDsCiK_2=&q7-$&E6ZcO1fA{d^al1&>?;)52g9<@*@dH5Yc_s<jUJ7=Mvl=RqbU`p*odw7K5a>Ecf^x$hH"
        "DLKf&+pOOr5FNVbiaC$AP(G%~^hk|>Wpxd`H@7=kX6NN3u~d%;{rTnP55Hx2LemT!U0dw9$r9w|bTrMwXEfAjSI0WoX?l{-;FBy`"
        "+@SA5p7o-ERnhZ}P>#DGXyT<Kr}?ZFTcCkcze{EixQnlzCP?E{a}6X-vmL*2x07Xkccy000}|w&;!xu*-Z`dRI$PqADujx3LU+tS"
        ">7|oPoq@MF^~nBD@T=9`6QTAL91S9=wSdNJqJ>f^<+aVq{RKTcy}p`OcRztmb5~^N{$et$k{!5Pl%fTMfptNt_jGYWKTK*bo;%&6"
        "s%7<eb#E$PXB7@%n0d?<!N$OI3kv&PIbYKwEYyB+j^_5TVe7m1j2E+?ATO~^^Y98K@b~N{ojxZ&;cF9%4#IE>&^uDXz5-nyK~6N("
        "KR*Qn10$d|{OR@<+?wn=80Us;(wOy!ghHMvR$Nj<pG9GSBThoXcM$92x>yWXXDxgUnGwC?37{rWYh&e)93RG*e118TK&hTEuv#0^"
        "c}A~>abhK)C@hvHH^Ikh-?0j;HTY=iS<EMyd!v%=0aIn6(e-&wo?JXTS|CpPw*$!9+n9N%-S}h(FA;ou#zD&FJ~86}o*FXS{L(0b"
        "14eKjj!&a#6E_P0F1Cu*(NyKuu^`XkZc$New&HSGAZv&P?spnad?YOiexG=J5>O`%tr_~XR{G?1c6(Xx@$QG<1C)!OB4@boqMsa?"
        "3B$WPQ~6XW$7oF{tSB&XCs0e<#ATn*4dRg}E??6O&r=s9-<LrpZ$q|6!FQ_k1``IE24GF}^!A<SmMrtb#v;S!3?aFlIg(qezzN!V"
        "X|rgt#LqD}wb;ktsF%(%d{`?v918D47(|nk3w|}bV=VS#I=K00P>v^L-zT3(oytG@5KnsBR+hLk89prT#?Kc7$GzjZ_xd~`{sT};"
        "0|XQR000O8001EX6-Rh_n*{&>BoF`qBme*ab8=%ZVrgb?V_|G(Wo>O?FKusRWo&aVa%E;^a%FB~WiD`el~-GD+cp$_50L*r=t<fX"
        "HBPcNGvER1;0^_vc1VJ~6oWuZw9Q2#HIho=729v$Iix62ikfr=3`jf|p3BF>Pf-+Iot`g=;o?9P6{W0C9vI*44WU}IdQ%<9Rw`nc"
        "29+IYI4hV=iIAW`er0t{$%Ynp8!1>8MbYum@e!{ZsSGhv75nj9CIq-8LZI4K$(n5VRu;6{p`)yjV#{}=v-s`$)4TT{);B3xKQ>J9"
        "nhA5$u;MG0M$y7(Z+s<7R()zs)0&jru)ogW|CX9!pORa7#{~bImDjAHT&*8z-BfJOYe@~dt||)Ad(Y15mu_{=tj4`{2E}7#2}SFJ"
        "qD@mB@&~EPycgz;tax#F#!t=Ej+Hqz1`5EX7y{Wc#YDlbtisP;KIaOG!v*Y?$<3dvF#MkNK{Z&Fyiii>92_Xw9I!jbM@0n)Ah!UF"
        "VJD0z79xX(wqnWh_=v!-WLuKwuz|_*ShH%I5)F-7P8yezdx&V0YbgGbaAC-bEY2>y7yRiK4B{l~y(NQ;Z>`r10)a4~ADVmw)CJb~"
        "gG4nWztgH^Yo(-$Bg|j`GwgtAYhv~ckliZM`8z3@^LtMfr^4tg3O2{AGTa_w))s_b^1w^8*RcF0TA#g*mgM{_CDHl<A1^Nd85FyH"
        "@{-q}?aV%h8D&_QKL&#?pc#WsGBvP4`fU(q7#4JE?ZFlg4M@XyQb2)2BQGCUr_;QBATPQMB7VVTRp)y~OXutZ7mTWSWF5NQaH?1D"
        "Xr<ZI&GZ<7zh@3vJci2Ca-}m0qvYolx>P12!&%-}*tuBQ-pvS}_)eaprlJSHrY+Yk%xMin83x1JNk5=)lHI7i7tkAC=Uc`~Fnpk@"
        "TwRdw#yTY&ec77TX?6-Fw^L!ul{Twe)w0>x!UBFRw0M+2N)`w_x~W($;7WnH22)E*HQ(MRqi^-qsrO9wV@Zgzu|%}4lX`VIBe}cp"
        "g#KQdIf|KAn%LtSZa8v0O;+05Di=^aSIq0Hn~(4ko{Izrv&&*jx=wZz*LG|<y6X~h@>6&dE~l`<^A|Sp#RC%CUHUaHFk)WnWO^`E"
        "k>9aH?q@(v_1)^<v2ov3E1fTlPhA@d48%t}IyPNMer*V8@M3ogvYo;zo=1q=G@S$1^t69tW`nBV<6DYQ$sIR+^4++Jr<jQ+wfeqK"
        "I|+O#W?CgeP{huCI!6S0nG3q28z3Y};`jkVP_-gg{zT@w25J@0U;QP!=LKB*+Tf^&il&V$JWF19TmpUkBpA^L0e7?Eg$MX6B$)V+"
        "!X;?-z&%L=c_5{Mo^E-CgcP|N0hXo<shD<D6rDQWSyR!{D%UXo;>DY;9+qE0Ff$Yfb4W5HV;6R3z#LV?R(u{xI?pHLsL$hM5v3#s"
        "R<OYTIs7k@ZsGKYl)S;0#U;G`h%2v?@G!mX{LDKCSBux7<4gN;I(FL81=~<NPD3^uC$Z1*Y4#3lU}Nj%j$J20b_G<@X=D{32-z8a"
        "BOpTy&8;x=4fXV{p^DbbFomV9xHhrfBOBIZ58|DoB?na5CiL<Qm+gm2zXSv`^b729q8&uo6$Nx~Wg}%3JKS)+MIy@LAo?UhoY@bc"
        "stQKOoSBa{$xs(8F+{(J5%%dgKOq-(O!m5k@lbm%+ryL;vW9@Dy%ie;bi9DGAn*eUHzpY9tKNA}WxM15W!NQR{Xqr7YOgc_RSF;&"
        "VJ|>7U~V|_QK;w80-Ek26uCK)>pg8y!M3<BDT*>H3Xvadhsl6dh#<(Zjz{Wa#%K3ZtOPR$rO`E(b)UEZ0Vf^ZD4leiT1R5s@SQsE"
        "mjAa~JHsp)8jy-zXzO?YE07O3Zpm}#9u;Dkjk()VLjh@N!?i3cRNg>c_BB;^IeTP9Yo2Jq2h4v^O9KQH000080000X0D~B=k39we"
        "0J{(X03rYY0CRF<FJftCZ)0I>XJu_|VJ~fOWMyn~FLG~nbZKs9E^v93SKDseMi71HR}9pLN-Ztl5+hLJB5|9vMJ^5O)GvWSFUgU#"
        "5XohCm$sFn|K6D;mp92tYrwX|ot^8MGsDSb^0O)|K3FJ)w$e&f$YKSeNa+`3mQ}`TVdWh{iit{5gTZWh@b((eSeXl}G%V@&LM~Mz"
        "vR^UJ#XJZ;t<fM<R%WCV4WtSRWg%076oZq3SR|WR6$o@)=4FOC7M4ymmdZ#{yoE$0Yn%sHY*JWD=j5QbOlt8V$*L3qi^ez0k3&eb"
        "GN!u`x`1*L$>1L4YHiItm`o-?u+%DtR9KN@!Why>jwo%rN413AmYkO6@E4ib5dJ7xwa7yFRFU&R(43Ov$vOyvmz@sx7cNz0F}<Yd"
        "E`&|iivZ|{yr3tVvQ=0!2>u|$JMuSXSs)225T-8I!ff*#t(J)o{J<9nA26xduel44Ei@vLEXvA8GBu0Z#rY>JjM4*{He*pP%w~^Z"
        "W9O^Nyv#S1wN{OdD8op~d{RI6g>T>-J`hlLE~Qx7^*3(?KL3>32JNmCx~;%Rk9`BCD$7h#qOICs0oJK9F72gOR2w?_4@rN?NR$ax"
        "X|2*GD@%fNmQ)!*H1av(W{Ng<kr8(UFf(<c{6og_(nc8;3y{<n=Wu*>`s_*A3aqf`pFTZ4>CTEOk20}EZT#`+=#)&ZQKaqpv+kIm"
        "N2$#H(aF=Z?l9g`e?=vEPH+^ROf4~kgP&Y?d?wZiVM*<7Zx-$43A0VkI3AKo*V#lXs>rs~uf|k4w_H^avFIF@RhIPx>*(e|tNV4$"
        "1g52ewM<hg;3JFxZx63S&jn;IA&L_Y0bQ()1hb52jY&@3q2|=gYxeryz<L{NNg}OIzCaO}wa|`4DNZ82R-fY^Bn3o{3<ptV-0~#V"
        "=h@H^rRT^*6-&9QwDYYO?kl}AU1G<8@tkh(uZr@4Ku=e84O5)2<^-e-Q|{@>rcbpF?`Ae1C~ZOZu2!Iy{0PY%Y6H$Qq;k20Lz*7U"
        "Qnh}9;(l#?SbnJTn5skyDel5>jp?h0g)i^;b_d(;{sy-PcOt9MA$N4C1M_b+9_TYU#?>krBwWzJg&lxj@B=5zGp`v^)02AWy*WNR"
        "eDm@6Y}PByCE7|Cqv~8WL(|AFJY))ZFE3z9t{bRmP+CHSsg_Im0JpbM9^c;1>Y3#Z>bnDdGzzN3Hen>>l2{=FOOkTq5M-4V8C+k!"
        "dJjY{4tDnnkT9|xXz~~_q>~+oFQ`@HCwt(&w&v#jW^O;rWe<Mv0*)A84R-kJFc(@#3i+B#_F8MDr<1E8P7UPL4iIzu6a}taU(6@7"
        "zC9v`X@{q9fR3;&dpNG=3|^dn)1Seqi}f8_A;4D!4Sp(P&rFA?x?&z8oWV8@*JUn<f$mwbhge<fxKA!NW>@%O5FdT@0n*1&gq_Oo"
        ";LrW!84H4xEO60O&H3H+b=|uR(gt~5(f!zBl8&5Va&OwJ{g1xvv&QGTJ_9#L0q$3x$E10WxkH#9ADuiQh5T>UkM^UV&xR(rUF=50"
        "YB;RXCi|RO*6}5^4rT`L*Ql`zI<8mS6%#TK?ly3b@c$j@jz$;lBX%B**tPYZMA4IjeJ0wAZ8t|{!qa=&YC0TinmMya^CRxLu@%l0"
        "&{@#a>?q{#0m>(Xdw%1|<H=&IF&y3-$1vEK-2FXKj&$@EEEghO@*f|O>}98=R=PDvT6guIukL#qdD)9)Dm5ncjvDdIbsKTVGsMB|"
        "E*JLt;Z7##_vZRrZ_n!Qx`Q3-z0ez8T)JI$XUqc)OqtRGBx$JAzEUEAEcAZ#r<orqf<}Cr(pq!qu1dQFj=^s$rI7lbX|!V*KAywx"
        "wBi&Z$6iz>(K04r^K7Glu}Hg}ny2A&kn{qku%2V7G$PKBcIV2_Kx3`sMXQbmjbVRi-@2DYs_vP%jHJzE{F=FcbTx>rlSxu$+D}9X"
        "DZiMU*T=C^^tM+Q;=KMh`&#+iSL&!Cn23lrvO=%WcEx9M-hGqmVBpw{1Ri?uZ~$<xor71NaG+wid0(*_<E(XiAt_XC>OBkcH1HJJ"
        "a46Ta_Iut>Ui)g9G9CIrJ}*L8v|iur@cXWG7vjyhFg+7tUqkROP)h>@6aWAK2mk;8Apph#QCr>=002l;001Na0047xV=rQ9W^ZF*"
        "Y-eR{ZDB8RWpHnEX>@OLd0%sMb!2%iaCy}{Yjfi^lHXI6|G?uXGLhuRWRi*Zv~`=rvp02@nW^z)>#7_V3nD>@V~SJ=(zY_?|9;&K"
        "fCNF1lAY{rol+BvB)ZXP^czk2zW**w^Q_=r#doF1>ayB<qHdDCw}U&ZVs%y)GjGicF28%MNW22R-f{28_rG`|E~~6qdq4j7n}zTD"
        "Uw-lB7uj}KR<*~fH6ReYy|H0plV_{;`jaS&_M#N+SFwlhX;p5$xXg1N<Fvw44}UD1qUMzvuwyl%)DPdoB~n1oR#{rcEMN2OmM!A4"
        "NVBybclkf>FF(Bb`SR_@=<VNrzP!Q())9Z$@haQ$q8`CeS1hhuN<IL8vg-1IZFf1J;iY2W>#To$Rj{4dluoP(t0ijwP;107E6O5+"
        "A(2p!5wdO++}JvWA^7qO4?b7#e);g@W%S|lA8)SSynA~!lehn#)taO$fa2=5+;ARs=-?~tZla!7oF$Pcn<|De<-KTcx5H~DZX+ZT"
        "pTpyDl;pa+<wf?plqf}XUIP9**nu?TNu);LPUC+GPra5KGzCgefct+z+a7xJzdCq+A&JCEy(LN90_PC~xu7K>>jPfIWdd@%^qV@J"
        "zw*OSudZ&Y!hu+^SR=pokq`dOOkJ&*;89#|w^@xh>jq>b=!;13)mJI}3B9zeJpATmg@VQ_;om44!j1*^RaSF*>2l6cEtM9A8(Gwg"
        "?QN1(fxHmQk5vP5$saONN98SDh6aqpP23<pCCzpx0z?f!g=j!Dn257%`I_Yd5M@OIXqV@q_qFH$TKG81Vxo)iyhpi|F!5&+-))TV"
        "FFuu7VJ-&+)3F-82HFc4gIl3V8?I}xElMkrY|TY&3Z+`seO7NIo#l=ffnTls&|?BDVv*$hqN7-eo3fM(lmm@bj-7eF*s$}L-}>QV"
        "!ynW$gOrN@)?~1_g5O7FQ|}rVJy|NLA$E!&B%{uXz0?ZeWQD8`Lz6O=3GSiXUJ}q{Rh3oXzoRKU7!GzCEfC95srDY)G?szgW?t14"
        "fAQi?Spov)0b!mbY+(vW%IkR3X+3O?;6R$#@puLnG2U=3GGBo@;V`vxibn7~?{@VZ_HD~+hEjXdB9q8HkiGp3<_8<&ZOeq`-aXIO"
        "8=$_tFVHGT2Zq?lY;BtL8g@Npu(g|}buyb2uObbE3qX)9E3y<eqi<$91_nrd5Jj}7QK+QcJMnzE)Q;Rpqgo&$*ysr})rulfjl#=Q"
        "4v1s2CfCTmMFN7H@ahI+Jdk$?4c-AkFmHGqIO<*Bm>Z8?F|K(6>(X>xx05W0k5QJ~80N5*g{9m$Tw}%VefT>BMDLXYhXelvEQlXP"
        "dQCl4M;imphJtNjgj5gYgkD8!TT#llAo|8i@gp@ED5u@(Kf{l;`N42OGc|X`?=k`IZP^Xcx3mD$|HJ+y)}?Y>k}N3*t9{KyV2y)~"
        ")J8+v2Qu6}8{z%iQZeLWxGt&Q|6qB;m1ciL;yjW~LhWrcv1On~3*VYM=yv>s79T2eEGZ>IIz?~8jHU=6L~Knz$|0<@ABY8Zq;`F("
        "y&|j_!55a{5XRyrZ3+Q*&qz`gY8;1gsHqv5@QSkb-j)TIgIds;7XjTP(Mw<=nVeKbs6FdJ!zmn(<l`eE7HL}n4*@GnTX9k5cRXmL"
        "nt_v$f7Ek<VIiQfBSlHjI;$n+u%0|G?*UDHUsktEdfL0C_o@KgV&-kK1Y!o%jZYjS=_S~jnfj-kIdBCdDR`tJeND6AAda93ybcU="
        "he5~#AY|mSdC_b!PvL<btV+u9ThFV35D^CNuZmxQD9SU4+B`s>gRhpPhwT7*XerPNX1ncf)z#h_fPpd4J{BsgeI{yYEkotT@r#@l"
        ">xQj4YO=rE*BkH-N9i);6OkNG6Wa=m;eg`%HBE%yZrUa5XOU~|?g?8Ef!PsAsf2Y*O!GmUP&2TQU_NmhajTD-CV{XT5+yJ-1_s9@"
        "lD|rlg#$G@1+-kSNe-irAIrK1EuS!Q{0|@gPj`&CKU8WDaa2E4bD;+$(%TOoJ0SE@P#=a#QGc~yllAH>Oz7+#yfBfY(zmE+%@z)d"
        "r*~C$2b%x@qdBC5tB7qjKE2~nXL5(nR(ToUg5{-|muMxV#e`JMd;&?81DNF*?3-9>5E5N=6VkUKz?MBkDr=XvUP$Ao?~$}yaJFZc"
        "vB`2Q-4FDY1=kM3SRKHXy@wd9-Exey0)@X7EDMd1NsjLU&EZ)7XSWc`9QdLIC;f1M(^xY@Hb9KqV~{d}yYQ|*`$!*rOBY0_=M(>n"
        "nr^A@k21d|yP_u{46c2BoZd5#_G`b>IKblHz!r3zKS4uabKWvX<WLrD2bo}k`KZGfD}6Fz4u_tqQ7(d9bUI>L^2VAHh0c~Xjw6yJ"
        "u<f)p&gDP8UV3L{9uf0h?FF6vo_X)C)R&JC?SUZE6=lX8zgde`QG+J<31SHW3AuTY3mGts3^cvS1v=sdgrh<4_RP2quEu3e<eZ%0"
        "-h*V<RY0o=uP<Jno;sz@DRfntqe-=kc-Ka-aC3dG_SLMfv=$-oYw(!`Nn!4zWJ#PLvGl-~`JT`3!1yS+lw-r0lG85OmeOKuiE~Pc"
        "OUaT?RtW-NjaEm1Z5T=k_0rrpU#<)oH@R?wi#{$2w+sH5w^@t=$8jf(K+YVotIC)|tZxkg&BZGvZaoO;LoPtY<2ukX-x~qSR%2w3"
        "!D~BGhqBeRmW!K(fY1em=mMB>7D8Ghvv*l|k`r4qe^QljT&f&lKJR%R7&7UO{NEZ@v7*ijt`dt#Rh2I6KxZTcQAubovgwfvakL7*"
        "Tg?3`Sr?F-a3DQdLS<n9@ImeBEJiN`c^s&HzK^ho<J${o+pHj<2q79|k@VrlIt#RfO`bz)!m6l?99fe9!%U2=;S_PXL&@@g`S|)3"
        "WnDCVB#%EJP33Vzvu+v2IWngL_T&*HNAWEvMZ^l=6%Yr3kjy^yOpmtS8aiYgy+l$+q$8OrjnReADl2-mTKUvj-*hD4n`X-jRM#|n"
        "7-Zukb^B7D7DXa$Pb)aE4+DG|=S@Nn4=TEf_($*?x|H_RQoz3?Uw7otM#k+-eA|52DAzGnYaUHk4U4<tsm#j(>Kd)rc31mY)}!lH"
        "vtI83Nlr3pI((Lb@fia%!cPaNjKl!eFbijaB;-(jV<Zxj=O?6eD?ql{vg-E4l*_rwa%YN=nP(rO_?qg)4(x(rQ8+WmFutNAuNpF="
        "7`DC69=v+PJ>ndSWX@62<f&9$dIG+Q6rMm9l{SThLjeJyiD}fr2=4GS5U}?3!CIyxLh;KyiBznLhD)x%5heI2X~sa>Ro6H~NHHH;"
        "P?I|&IFB7GniaYo3Zhy*S_AC>TFUg&H#gNpZfuS!lJrf2#Xpi*s+~aaWrtG!DqEvKb`rE>@vXfS|9ke$*>~Tbr{BF`uj1ADi`DAo"
        "tFyD!cQ2BcjIGYT{f4J;`r^A+13<)R-<(QyoLS5`>rj4haTVG<a(TlQ{`}_SfGje$dhfJh{pwdc@A%pXTxkrJ+wH7So`Q`6g3kx9"
        "$m?eJB;acT9IOUyv%Q!R+1NEJaM#bNG*FeHD=L_7U|2GMu=l9U-bthKLXPCE8}po`U#{Ji=1pXPN2jEtbpXSeK}<42gtdJx^LAuS"
        "z^Vs~RsbB?zvTjqB7n4|k2aZ)ZXxSu=5ocx<EeDn%!T@lb0k+*ZgmJ&&2%!6NI5sr!ZjLHb`aL~b*Wg-e9x0I7Oi8pKn^BdN+yhC"
        "=6Yb|R3JnrrBBHo2pHm<x{S*l$yR}SlBKDN(IS>Gps)H1o=VpcSmMF7fY5BU2c67wd_p@fHazj((;{CizF(ZZpiU<aRvLJ{C)JK~"
        "1_%<92U@bmBJ|H{8wLzIirJ5@Fh0rHx+n$aNp=vge8KwnY^Ou(k;uG5(zZ8g{BczZfo_$K4#ccyd4FT_u10@UQ2hTQyl+np)Pe9)"
        "TC(4mdEzo`E9wg)a-NRgx4<-EMD_h7dQbrD#)&2cDWK^W>)$QsO4S<+=Tw`3A(<qdlTKA{t+gcC4`gurc@EP$d%!gp`v4Ski|J6e"
        "COozP*52beASFzCNZ^+hi@^uYaAVX7J9QAiRpf{E!T#{@=!1rugMeiMTO2B0R~c%amM=QYga<hDmSKNZrzpL1@Nr&R`eU&Z3{5?G"
        "SotUL`iah|=lGuIoW`8vKgT<nr>E2ING<LE(jAqczi~#KYW<RIptA;UQTAfA|I$SzU>sRoM<hDbcrkFcPpbJn*)?fqO{}5kkjOtC"
        "QAk3$X=L`S{Ihg#+VMy{>bRQnAc)@^fUH0XbnOkyX_MET-zm$s<v}mQZhB~IFm9bA6jNdyww&Ww)TO56F=Uz$JunPgvu=W`cB|Io"
        "kjcS8stn4bH6-)0JI{s%3uPS)4NcrtgEiA0#bC$;$EIqJP<Qp#)|_PtHD=Hml6mrRV>U+`C_C~|YA}rg3?a1}QG}C-VIQGM<WIU3"
        "Ya(r_S?#eYpm#(tqzsGyDn*rv^Q^le`?B<HNLs^@b@b}8M%UZtX--J7dTg113?QOIZ{DJrWWal4dsN$`*A%x!d0)t)$eW)!ezYes"
        "Bo*>2gi0zu$I1vU*f&{ze9R|R&v>BqOU*o#&4XU{rxuwMxC}c&soTBv+A}Hl*rbOAb+*dsbp)1GJr~=a77eHjbYGFFaYMpYFO+V2"
        "b&@`4aD^zL5%_{bU3w)1l;q5eFhG&jqttMC#>7_j-fe_$FT6XFuG<-Wp~^6!SKW%ZmRoB7Q3g=v1Raa8eOQO$G=OT(UA>v)sg+uT"
        "1*)4sn?W2m$o50qq#9;Rb`A?L%(=ACb=jKKt})?oRHp00k{~W(wgWBHHr?n5H5oqL4}+K-;xHx0oge{9$qO8%%W~adM-NR8mMj4&"
        "r^M1TBUTMf?U?&~d*~HMT59Y$hNC7rY&zOq?KgO|Z`1+fD%;h5h%Uu<6^s?JomMaLEL=J=^$7OBb%L(VW>*}Oao5rkOBIc|7Ae~h"
        "DR^qn>C6Vj{u9?~WlKUNAbl$m6=<uQQzWaUfvKSG-;QZ|&Ba|-l?CR!zP<Kgp4J(?kq9VIf4ksB$I<=7v{lBjNPDu)yhqJ71fCl#"
        "Am3O=_f`O@N>sDK2tD~H;+PIMyGPe%M1C`YfXdhx2eXIAKJ~k%+Lc1qNzJ!KNW)3bs;G;L=-^le5%kP=8xYk>(J4@*-i+(s7QsJ<"
        "ebJNrP!~(ve&{G4<nTXye}O?^!=)GE`<c3Q`y##{dLx|>sFwzh$qBMT_sAP>8tU?uMnhI_3<K>DMab0J0<NO}E*yTSD3l|NRw9C$"
        "StCT+mI*}!T!>ogx)f@1H_;-R_wn_gNLF$Je__Afal8z4fg+%<vy|RO)Gyo+ns)o$H8vuuyH;Z*6F?Y1t_pM<x_&lfmBCD-i(b88"
        "BD)@TCCHR|pbBlq1E(@5W*fuIz@#M$3_t8lH;5GYz>Zl?QZ034)`9iyr+Q`_rTfmyqx#O<@WokjneIqGqT!qxyY8n_fIpyhMtvE5"
        "gP9TF2k*_WQFuq<<_2=Ln4<U$Z`%slYJ|?g)Y+-i=hGln|8NHdj(YD%cQ013%@SMx-UyJE_hVavHC$cRAQW1SOlW=Yh=yV_dwz0@"
        "q}BO2$au%v9HnE#DlOK;u((cdKaGs0jk%CAGcp7vDKeTT%iSolR?-#0-Hqd$EcFL>DJvS|sFuqYyJbg0_oBDmkZ_1ovK3dBDhHKC"
        "Iu^HXI$C82hiYG+X(`vQ76*eK)Ja_#<Qt~iy7CH#qAJ!{cMq!x3Cq%@;+9S>k5E*lSIbGIe~wZ<Pa%6*{Q*UE9g|xMNLyp}Tf2_J"
        "d&;Lkx_d4NnZLG!Wp%Yb?s2fDKRIS~RryAa8C(^eL#GMT(bmLy0&xy_(*+#1x@tW!QkIsuK`#=EL1=KK&)FITW?gZf>>--~kBBZ7"
        "yF^i}M~;tm5RIo!$dbDyR(rem<SRifefmc>7JsYO4aRiu=}C}qnVw?`8bwJNN0E*Un-!i;(J<`n2Ue}*l|ZO^Nw$R*)aovea})Co"
        "TiM1+8R?4PC2-)=+m>_H<=Pue!<Zs4@aOYZv*(HhUcKM(rT$5P;iM{na?Glp`2B%{8I=W|;c<T!bFq}a$YA%Fhm+F}uwsOFH1R{4"
        "+On8|;sDG#t6e+;NnxPSbQSng5yT0;sA<&R4xoPSrVJ2CLFnGd1x$>zWa-ljWf3^0gBwOwnz?$7^jR3FX?-r6)~C27M~D-nlZ-%>"
        "#W*mF<n{BJr99DGp5ZApltF1Kzd8N>+XG0(fnS^-5@B{f0e*IRdU^l>B$IPw-2|3b3WZTMA64XeH^?7=9aif=cWGcBm_B)@#?2Ac"
        "o<M$_c26L9vPMndI#$jmP&`T0+9srlt?D)P+HlWg=Q#Y6xjByNSZ0o6coHvV=h%>svYu!T8fup=sUD<t6kdlY4lgqTLxt+OY<C{D"
        "n_`cpBwgAMCoxK6j;c6sOYW04-Hk>UT<2JDeFnU$-s?rBdC5#uc{&Mvtjy>j(Xrq9k#7zC1G_$oqO0;5)^Yf|XHUYZa-X?Tg8Ksm"
        "7E{O$L9623GohbsN=7j~$qpQ3?1T$F`~Lh87D!wnV7iy96HYWQ&kkalSji#KV^svRwJrytKZu8lrFUvf|3@@*3aU~Q>WA#*8+NVJ"
        "{mU=IjMyrW`#;F~=-;tEI=?*s0sjq->GT?oC?(1t`YEv#PRD+-Nc*3!qmS?Y>+-EUJax5$h5wZGeDeeT`xmc==dRt<d7QP8Fm#)#"
        "BP>~Rv3{pz=K772nWGgh^#0vDJL*?O-WzAAt0q}vy1Q*e?X9>+DMEI~2S0)|_*tVP7u*k`hOGn>uWBAUgAq}W;CJ<qAHpt!7yIY4"
        "`aOUqjzOQq@OV%@pZK8w=T8GDGr}LTIyj$ssozyl)lZqGEX#Rv;e961$6wnY!%#oBcPh&*u*!%kXHi5ej3O+)M3G*kl$5KzsQLC1"
        "XQSE-Mn3-wP)h>@6aWAK2mk;8Apo`dk8j=u007w#000~S0047xV=rQ9W^ZF*Y-eR{ZDB8SV_{}yZ){{PaCwziZExE)5dIz@|AElH"
        "B!FMKZfl1Y0hTmwkv2)Q+1)S<gF%xgn+qjsBxT1f^51tyN|xf-iRJ>0NKt(5^|`wwNv?SYk_nSLZ%g5o$=OUBaqAe*C4;*{E1^77"
        "xre#s@x3W?$*lr6NRs66;P4=|CS#I&E;M%z98+0gtZ%mH(BZ)^oAnWX@fGrui7nyy@F485vr=h!M&NG_4;cQGe8W(zxNx5P(zQ2C"
        "X$7=4iqaOw)jRa>?90XF`Sk4a^Se)DcEXP0xdx}&pkMOt#rH{%B`CRu3{YK3OEgTX=7VBQi?O=vXGVf%t^lD@C89ydQepw(wlHWX"
        "R#>xyoaX|Pc>i(lDd6*cz}|A<U4{J{pl-?=yY8V^I34b>8TVol;0p^<MfYvK+>ef!FLz*)dxpYj%@L=<y|LiUE#&GehKUD`dfI+!"
        "tWfzp;)6?`Q}Bm<VAa*Zj5VOjbA+042g8dAtHEsyNJP%K_Xyt6wm;IHh2{blMoX|8XfAo-@!AS@V!6smCwl3?v-*92MTc6=3&u;S"
        "d`IM;l3r{5K!nzQl*RcVMr(g!0s8206oI9Hp>Hv*xK?vAjyKF$HCH*;(Ll_kDGKfCE1r|>)jo%~6}U|KMP-#sBM@Tz$EY86SwRl("
        "OJ(6<tUVEGS1(@EO*#5_2~+f3xHzPd8^+xV>^dx%MT48HzXCO1c$R^kqsA@%c_!MfD4kks3n^HjWNhzN`0@5BMiiQ%j=|Qkk0GI*"
        "NqxredhElmry(4;TW1+~s{|$cKhYRZ*dB%2m2Z_0kj+3!jJ-M*$D7+%#$s-BizH2f;1)=`82Y;*LYB4E-2iO%>R3l7P{e4J10%xD"
        "W_leIn{jcw;#NLu<6UW0p<x<%g^~IotIw)*uD!~Of|3obYuE?KGR`z#W43k5tt-l{eemH=h5EkKy1E|K#}~HpIFk^T58|pD^sD{Q"
        "g4l7xn?_dk)PYWi!Wt}bpMDQ!wi_~M;hRp-lbv<KSG+Yevk-?%sH5m6<f_ThwXQ6{E|sH7ALPK#t-)j6a9{3Fd4n4speYw7RT^<_"
        "M(-!g6kcUS-a+g76Q+$RIzs>O6m${5@Tmen6~08e${-`l5+qhKbo?5(xI+ddvvmuxNsMq(z#z*`_lY2t^NI>)6Y8I<ZvwcjYbGj9"
        "olz(7i%t}#@@jWD2Xp}6#a|PS(eC>x785qd|KyrQ%8dzt8oWgTsaZpLLnyKWa@y}&NnSscsU4i#Rj!HpZ868#+Yy(B(%+Ab9dkJ^"
        "+`@QjLa~{Qh4|kR$q=bJho|WF{Ke1v+Z$Zo`I-q^_R(;ssM=L0*h*m?T0Pg?j3`8(^%X3+E`x`vk~b~Ze(~5}7(0EH*FL%lXP6Vl"
        "f_;mufzr+J?nTD!Et_LBGQ3yG*oFQt$4$1{lYNfc1bXAVsY@SqD_5Zrlzm|q)c3deZHzRaa-mBZnyW~=k(H>$qY~7yYdPJ@iru-{"
        "I>0e6{E*1zQCb+IsjNDZDbzh0X+l|8f;LE&{qS*Qs!r04B0KLRj8*_(4M&}uz8;-=j+Y@tpxL;xtU23>Ek?x89bDNGp_vxkqg2FR"
        "FT_}Tr1&`ge0_;^%r<rXsq`C1TWp!2nUGqHtrd5D$5UtzP>wGF{l;S(g7aaV;HGDPPA|@Jj{6oS)B%l$;fbb#>IsvlJ@nnjIWW$b"
        "?_(Q%Z9E!Xu1D4ydxWct#$aLZxEeokxIN87hmc37_LwxHj3?_U7{{?3If}EC&nyju_{x<+poh~^Gh62MkW5!fNZE8s9ny4qM14Uo"
        "$~~<{T|@xSl@1HkG)Zb<=oeKyVQq8#Re%3?bB-<^)cbh8e%v}yD--GCgry1YxM8&4hc6fBqd_LW`&NHF&}Iep==eTq-9wGV(eYrT"
        "Q}c>`X|nnH{!Q(+YD~Xp$_3NaEf~lF`xe!|XUX<O*F?$AL*ekE$Ktnz!hrb?P)h>@6aWAK2mk;8ApqM@2_lXF006xJ001Qb0047x"
        "V=rQ9W^ZF*Y-eR{ZDB8Ta$#w1X>MmPUtei%X>?y-E^v8m%`pzbFbsv^y;ARBd8ZdZ-4GXm4WY<&lVV8X%1M#9JpqZ0?G2y)hlmF;"
        "&vkb?o<VZ;7KnY>L)c=3Y9v|m0_Ia&1qf#4E%^rLFJFwRXoe{&KL$kXyS@vi2_^~36c?Wo4ZqN!)4jdP(-)Ams#RF2^3AbnDXw)*"
        "*Vm`v9L4AyjBvp}1h0Mf22e`_1QY-O00;m803iTQ_x$jd1^@s*6aWAy0001Ua$_%IX=ZO@VQgn*ZEayMbaG*7ZfS03FJpCbVRU6="
        "Uvp-3E^v8`7~5{!HuOC}{((~;DhIB+3>yZx^V7Bheb@!quou@5Xo-r6N~AziPTU~>zH@kyNJ-8npf!1JJUn+M%d+37DNn2-b>B4a"
        "Iw>1s<QFFRS8BKvgo=ukvh5mX*gMwA@<jgn_&2Ggf`7dt9py^DH;<AWsb&oqOmDI*yS=%+;cX|CAuTnh+nZX+7KqKM;YZ>ze_D*k"
        "9eFKF+B~qfrJITxx{-B_@(Nv>z|lxr6=koeVO4?3u8B7E*s!AJ;(@77abd0sOfD8nJTt{>4kA4TQ%f?8qFlf+ON!Op6cPAU_D(ju"
        "yaditikd(4iWLrCw6bDwuZq>QZ%h%`icZOzH%z}OTY-x-H&R_<?ZUN+2(4HtRfXoiy@9hx2RI5^1nUNox$aoGCTQ%U;=qO6>p`R{"
        "N(%9_+TPv}_`vl)OQHLg5!y5aH2um1`Sgn>To5>qsSiX*qA5fa2wxM$-f37GKoSXls2ll6DkZxPuVW(#<tq88of2d>kAghaZsm9y"
        "-g2XX9<loj@2*UC*k@LEjVniz9UMaiCIJo6_GQsh!E2^XIBy;!)-s!c38Oimn#HTageQVUEID^*tL<xt4E(Phf*-F*%rGKYqZ!G6"
        "mx8Sc{_lDGkL*p?TU$N9I0z=Vb<CP@!&>V<bi&5&^5E}?d2h)NAY*?RkthfFklk`deE3218}@{z0HVzUGkNAg*$P7<Psy#s?9H1u"
        "<de`w^(ChAXYmZo`?q`qy*<2q1~?Lagq+zDM}hUnqTKe#KsyG2EInnFZiBfi(DHgFEol0DcL(RJ$#A7@xbMq)PHppS4i2uLV?yLz"
        "<tc1<?_7}YGY`!U*JVvOu(%f<l^y0>SEFrE1fwoU3k_nFxPH~{X<Th+*D+BI!C$3OzsKm>U<goAvVvDO@(^e%Pa^6-z8Q-G`VPpk"
        "960PGLZJ-89sZ$B&wf=(syzGXidrd8SX+6;gyH5w$`geaRFSHOCiIYW(M{%+jZ*)4JSGEGp-Vz|WTC2IH@2k>%Xo3V&mvQ079KS`"
        "V05>MVX<*D_S8XTX_=+9E)%q6F@>*$TkowH6^yEfb7HH-lw)C{S~Dx|c?i30Qv4t*LKtk5diyfFsW05yNgZE0xSE({J{(Na^m!={"
        "Aquu+KgZygKgB_|CQl*w(Hgi7nFRf`=1&1?sDk2#yVbH>jMk^2O3rB(vTvzG`#=v>I%Rqpw<~YGmozRXAp~K%V8KA_r7LE7HKtf*"
        "2M2J^2&vc^_knMmIdEzU5f5?hed7iox3<F8p*^41al2Zly8G>&|FyHK-rf7{5y#v@LyQa{T9$p=H#S3e-w)EB3T;@YceYo0S!dc5"
        "44U(0_rtf}dZ4CfHUlb#vrmB&S}Xu&w+BEaLN#O6?*2#EMZ?n{f)Uk=_kIY(8OmLR94}!c`~anfm0vmu&LHRSi0(Vg2u&c%VS<EN"
        "%4so9Pf#Ddl%L2ww$;%Z0TLS{z=Sl;0bv6kh^phfGJX<ZC_Z~I9Qz7_e<*@oLxlj3_Z>{-42KS{+?{<ZY)ZzPykR1@XRpY6l1KX9"
        "C8~zm#~p#Q-b6TDeQ+t^+Ri?%Ee+toMqUWq{(s$4b_QxCR6F7hbER{*W=&h(T+$&g)VxB9<`PXEs3m$l6qBIobbib9rEQr}yv#-("
        "Th9%LVDyzJ9P$W7T072+j-2V}=U*ec>kp26wRd;%HE8+8lKYT#uwVmxsklc7y!$qw0L7PBc$Yx(XmBEy18arwT2bXGz%Z1+(?{N5"
        "k4@Vq1v)5nCDb2=%~Js3nI$iSrS_RL!^niwmS|B0qEb2|;i*xYe3f<}P{wu~mD}~EDE#$`BBfMRXzcia1qM%)r|^X1NXsuriGver"
        "jnuZYrp0{-trM?cB(cdd=<RVE=ekWqvc4ka43Ep>t3pPKsf-YZOPYAxf+neurHsH5lxD?{8BtxuFBu|S@PY<z5u-2|CnLj^R`?(D"
        "*T^+P9wyHB$z{{T4mG$)-5rV@pvO&25|=<AinlQtwit{__(}UcHi!nx<>c=cQK^c%Fn)0-#&7J#_}O=<Ys+!yCfOUo$1dw?ZiMoE"
        "+x<lFsDUrqB<bc?3^K9)Zf5(a(T~drANBlk8R6shAFU&bVG;<7SUOGwk+ZCsF+xLh?IyF%D)uIXp6`A3@n8)Y+MpffCap2$4YNZr"
        "xW@8Knxvog9MkR1zfem91QY-O00;m803iT{t-9qN0RRBy0RR9g0001Ua$_%IX=ZO@VQgn*ZEayMbaG*7ZfS03FJx(RbZKmCVRUJ4"
        "ZZ2?nZIQ7~12GJS_ml1&tW2#0R|cj-VuFqg&>f{Y=Nd@j!*L|~_V{`Q6{(qQ$MOI9k1=L<ci#g|5-8gU(F!u)_k<i0W_f^20wGPr"
        "f&&l(<5vx=@%{*d22S8@yX%ZGZPPYOCkU}S%(-Vuun(S`jU%RHyW*_(6o>Or*rJun_(JlEb~rvW6VET(W<akBMM6|6#e0ub%Bn~m"
        "6ikybNa+`;Zil%L-+-WX2jr`_Yq)*<SLoXY^eFna&J6v$cnmHG*ciq<8?Z^Xe3p%1QcF&xVDs>+1CG*EMQhGc%@XpsI5E#<8wy_C"
        "Z?!a>#MOUu1Lg^R$T|l;s&4Fq8hp9?#h1VyJt>Cz7jZ7A=pRqW9vQR4?hLVRn;%e10|XQR000O8001EXY;1#b7!Cjcvo8PuDF6Tf"
        "b8=%ZVrgb?V_|G(Wo>O?FLZKYX>Mt5XD@AKZEtdUUvP47Vr4FHd9@l{Z`?NWJwW~gt3I6MtQFhMr3bfLz-gPq0Zkh;x!!|e2waJ~"
        "=2<H#B4vA9<iFp{kQAwpUE94o0f|@Q%y7PEI8qcvAGW+KS>0&4mA`W>>+0oizh1m!O<S+Tf>pI<`dZ*^(dD?zKD_<NbjxKW+VuIe"
        "=g;2Ng;3|Lt-Gqo083V^ZdrXR+LAZfrsd7GO4&ysQgW?D%UYoUTQS^t3&^GV3*$ihmbQQXj2RrZ66?AZY{m1NJKh!xw(2x1YN3d$"
        "yySAr^18~qwiT6T-~aQ63w9&gN|a38i@ZZ>z+n_c$fn#jbqm7Ctq5=HHoqR;rB%f^>`_e~r|YgF9q<w%-Xfjn&wvr7*n8C1PxY@C"
        "KY?O+8v`^brYl8q4idpth&9VHSxKE`u@dDP<l#!3!$zsWS+JH@H|I=(FiUp&+F|0Qx<<5dlJ*o5zb4m&$i7{&lK=%TFq0DOH(qw)"
        "$F{B8I6@W;__k9Tbj+GsNl?B>qktwdm?D3`3I3T`Nn7!$ldVu_<06aS@=}R03EF8%8;({~&P!QsVDT(HVeCtAhvu5&EjyZnFYA_P"
        "7qAZcde2+F72r$pP+Vguv2FA&MiNHFsjRZK;JO1|7hs25nTsVY+zGk4)~R^}OM2Icr6IB|Yp%aK3*;G#Mhv}q3`4(*wpKCiSa&+$"
        "#qjvQ;0Sm?$4INXg@85=W>vOeN*BwM^aN48FG6f#R9RQNMHf+_$3a|bG{@VtG#=gmFtCQ7I9Y_|8Qi@I_=A}5+R8hRC6cfgW}{Ts"
        "jxNcZfD?pnVnPl1hKOIw4d^eU6qQjb0*IJW$$}j%z@tl<@6JK)76gcpnP?JR?D#uY$Xr{IA&3kbB^StHe>l8o-IZnZIZ(|ssy<pK"
        "91Ry+NCLAVlIY`$7#&BCMO-fJeioJ&ywRcnmM^cyj4qvQ0c&7YR)H%na2LovC`l=mHrh#XsmRQdij!oNE`YXFkeW5GauFM_g-=ik"
        "<5fZA(^fU5)Nz!8f=-jm<I^kB*^nGw(I>RizoW&>(#`&*+x<)R?9%7+DQLs7Fl0s_P<vVl6$CTdj0jU~1!W3wtmZaCXR2Enox}m<"
        "Bp|(*lmVwhbiQO}6Kmg;H;t%@m<W2PFWH6KZ6UF=GKc&qH`Nv^orLNn?jY1^CbjXkR&2ooO<m?Iu)D77)DuFD2~)n8Y!eCw{w=B;"
        "ezlng0*(A!S7KOW0?+V}p?8cd;DN@z{;};GZO<@xc;m7txv{?k!Hg05N8q~@2=0EkgO3zyU)&Z4Blzb!;KlF5?gX?TkcmJNFp~>Q"
        "o6Z7%k4#WT=QcV6_l(f^ut2qNCVUxmK!Ow=A{ozUU9bYa^_38)XF2mM1ElJ<vDZ!2ZAA$Rn<%x%PUdN@cA7TUqEdBhZsQ5kXP=vg"
        "fOMlM(L&a>+E+rL4?=Lc7Dm?^`YGsL%q?<hP>DrD<UEXt_ue`XhODdPr%pfz2}M>>vnpYYL(%+()hl#CCJP7~sdo@V5Hp1+x<<L4"
        "7gPa}1tm$PLw7H%1c|E%$oj~~@lnc)BKFXuAUD#zSXxPXV;s2$;93~enF3bgp?eXo*KW_SRd9bl*0Dnmt>L+%GCQmoUKQB9_hg4a"
        "5c^sT08)CCnz|i!qzFz=No{+8hAD;C7q(&gXjExn6R&|y(`8$+t=nB}I;5E@ZYY>#2bE$=!M=ci%e!JwFRiMO#He+H(@C8J=Y>(="
        "aulBiS&8=c5X8C_V*e`HcmHxD_)Yc*HVE!8FIJVv&Y}-Eqbbx#3QEne!pUk~JEaYLX{4YrS3^=@AJ4ZVw+NL!;B|`Z0A_FljgmBe"
        "CDbPU4%TK>8ScyPvqvyKr|Eerdn5>a8Oq~YmcD<5%2jubDar~DE;_%)Gr|$}@1jx5GwSKYG8!JOp;nhreTf8QJ+V9$$FC9}5170R"
        "QEWs@QQ0jLj0e26J0*-_9Cl*gveSdL@<Z1&<&J@i88$&Lt=WSbM6Qt|Xi5+uoGcLezJ>yCh~xY1V1j^c_ejjc4LA-V`mX!%D8lXE"
        "mqMd?g&oLcW2)qVmSfav5WJDy;i#+e2C^XPP$U{thG|Dcm@v_APxr%423#{9BP7nMgWY=LVx1kXBoFGb#P$--9I`3|cBR;IR$xm!"
        "R?js_Sj@dzjl*M;E$;2HLtUO+4HTK!{M<{&t6kjKhse79HKW@@_BkxtML%xxjv^Gb2H>gGI4B@x_wMEK<q%C6rW=fVT1_tszWrBB"
        "27zd7mxF*@Aj^X1*Gn@YIqU#tgDc+)jpPqRGzuVt#6G4h`&99kGZin$S87+~*KJ*4=P^k7BREWEI-}%g(;L=wHN>&gy5&rfww<5m"
        "tvSl#3@(M7vCeXcRGgYvkP<pLp@PG%s(7QWYdv_@<+)J!g!Jjo5xdbT7;_D(4Z7})h$*p+PbddkpPAF7Z+mc_L(z}nmFYUe=vW4)"
        "8h5!AhjtjHv$+jA-E>A&GOfwXiT46q6_C^4<xHticuv%Yk*g_vJ5~kqW~UU@ZInRwqyfXj47QC@!hr)QHW-9!9#oe4SjFzzlWCx@"
        "C@8flQ&PAYblw7RL6iqWNq6;L;f~God;Srj)$uq4f1sd%18xM|MvzA{JO1)T^yp~nxSywlXV{r)VUH3xcz3IrQz6QYYMckQIe%5x"
        "Wo%+msx{8|1leFRyP0bHSBNc$wElow8T9>n#fui*$UQy)Bzo-=N+C3)xHw5+hiyDJ>OgZN=)x*#;hb#<iihDD(f(8r>`{?*_azI%"
        "Y_Ao<U~!I*Q?@<E3BaFhqjyrliD>h(dYr9-t)6JI5c6Tn;f2;SSkoM83*R7>qHV^tEdY7gP?%pO)MiRpO^Uap=E$^Tv=BtoeCong"
        "PtlqW`>8n-Zjuz25?iWmTpQfO3GN|t_GD&l>e@U=v&7aW<tu1wQXJ<b>^pYy-!g<PkZ<6;K5&eJMQxU=@VGCUIsZc6=!uO6B;!=Q"
        "H!1t+1_A6z$jQ|E{%;lh5zQ1{k8P-E4l-&Y@fyuz9uR8^_=LHPLwRsEW5s{ed<L%|=%88bN=&Y!mjP&L-QhSfdVlfZ!+3rlQ7$$O"
        "A;1LZ)Hy(az(3<9>hhpbrKzbXt(U1WSg$B#X4=OO+w}>f-cjv<jb2#M*tNz6H~s@MTdduCEShKbx>lbj@`R2L5dGBkz_@Qe2@IJb"
        "nc+-!%nnV=(pfj3?Bdh;SmJ4@kq(+r2TwT1joPjQjLZtzJo(%>-r#`|c=Mz){Dis^d0iDmb$SqD$L^psMl%&0V}pK-m(BSLmd%gx"
        "GGA-*_i0X@u;)@X+SmOa0|m2z%)GNka7>_MNdJXKhHjOaFEms}-*1Qluqe&~w7~rStcRp0V_TDJ>{%HsR+i&a;NyzJpE%l$lWmS?"
        "{?T~b9BfE}(S~2L?}1t1s=lj~^H7|fZTSwQ!tt5FiNuzX-PkX`y!!^o_4<H*XU$j8KjEM<9nwBLG;o)Rf`j6^NhMS@U`5LuG7D&2"
        "W#FG1oGZ0rE$=f()f;^crGcK}j0rz!Fs~GKeFK5BfDkDPoQL25$}$s0(OU`q2>0}DlpULGEI}sjd9KSH$S>ds4gE`3q1d1%jvxB)"
        "ZBs&HOJDg|y$AUS^Usjc#5Ieu3R+o=*boY5po3K+G!$zuC46>@pW_VS@yY2|rzc;ZO`yH*hMV?}=?(iAfPeCOf*dH<PMHtR!DI!s"
        "To&2HtScOw@^<H%KyeRwkWHj`0kR#Bv~^4c+q3J(QHIF3gD;b&JWYA&iVb;Wohkf64IDGGmdE$g{Ro#N2OeYHHS}KI`x!f8acs7K"
        "VJD=);xRi-*vprbg(SH=C#0*v`;Yi2u*EqT1Nu{owm_o<;R0M*wLoM^^G;vkkiLJka9`IQ(XS0kOmA$)ud~3Ze}U7c9j(L3fGWl}"
        "P$BAumB>)btktA2bw^)w?GPw+aL5$0>{{VN1YdF7(IhT>f&E6V>ztQ=7uzjA{_*S${tKI*5x}cA`0oI~*+T%YrdybmSA_hJkMm2f"
        "!lloH0XeDK4a9NuO%IZO8g6b~+;{+tOskYfg-&40Ox!~Om-q@-Ik&{;pj`UyCIC$ayb=&_f~P&dr*TKw63b_9os!ROm~Wg6uU{eW"
        "xA@;zXJ5bmn`!S&S0cj>-Msvlt!YbS@svyvitF7ic#nfT^2?JeIHRuc53+rEYHm->E%ucw;({;Whn*0ZXu$t3PtUI&7~qF&-~>OD"
        "Ro-Hy?7zyW@6<V|{St%dDi|8O@1k&HviYb_z<%a_!D9O@%z|_!ip=J|1n~sEyT^$>u{}O_TS&}B(`swULLwT{g$I8<JtLREPm-wf"
        "Q+HBd$GyKtiTOTch&bwyhqwI^jM7YOzJKZ^b0Up!KzIAcsWTsT;T3u3hdz$>*#~I9F@kPTLwdFO9grMbd+>%*d$i&6eb~@AbiABm"
        "KY(WRhhOZ_k9}Y!0S6HQMM`J*jq?`2)4^sN9h+Q>s^<C3Yzi;@fR}c<bcbZB_2!&Bk3ueSc{Rh@`0zA2)Vqok4$bC1p0i*vdIq*1"
        "Q%d`po}ssbP`w${)Uf!dyl!X=<U0ZC`&{nn-;&!GP^f7$G57=)mU=g+705E05B`LYLD0;-^0U|+43=S0ot|FkHG%uul-drJH8{eD"
        "?8EQnw%gh{1#XjIehW&>5#0n?A1``2UY3Y(>748;P~Xz2k5Upq(fcc;?yKE@08mQ<1QY-O00;m803iSxW4xPK0RRA$0ssIf0001U"
        "a$_%IX=ZO@VQgn*ZEayMbaG*7ZfS03FK=RMWn*+{c4c!eaCvP}v2KGf5Z#gZhpQ*M0*j;@RXQ|ErjAjT2t_WKV9A`3ZKUw`g_xv4"
        "tvBC$_wL?3LdYRo<sh`Wpsb)1zHCy1dVuYY!`5gmys_XKReJ{pOZg7kI0tR-JfQIb0&x)vIUyvD;z+ffu^z_8vRVE(BCmTay)q~?"
        "2nYMAA&x3phvDbaj`C-T-kNTpaRlSj=AyjIXN`kIXKVsmd`tbOIvO!no)uQ|nu9_Q8%S4K7Do)yj|Djo>Z+axaI91dFB^D+(Q}Z0"
        ")YenY_nnq>X1hjo@|3R&HsLcAnr;xA^`~crauSY0_i|^gv6Sqtoh<!$^PmQ&&*UpP79g-1<FHax!y}fUdZQGau*cfd{lAUTbe=oy"
        "@$8>*fdgfWMWpX7-VM9Hr3_vn3xpZm^Z!El4T9I&2u~Te@@AB#s{|HDNmm8C2lT{Z%Xl`=pD<+f_5)B$0|XQR000O8001EXE#n{U"
        "un+(Mmp=diC;$Keb8=%ZVrgb?V_|G(Wo>O?FLZKYX>Mt5XD@GNW^8G0WnXh<bS`jtwOe~{+_(|{KR~_%E1)>ZS<8>!T?;o`AWa;!"
        "z{Pj)<ysWmg}{}#yQ=l79=7*_eD}@_Nr{xW>veJ!a2Jb@;c)nyC$*mE{k80hL?rBo^7ak;`TfVUcq`(&s!WTPZADgzx?$_GX6;t6"
        "oVU=Mun%R%SF9*ou`0{ECVctg<%>57uUb)`mPNK>1C569rs;A)^Q*KdV0u-Hw&p4R*_2%!3m8-j#_xFwbTa|;Zc29h!;Ix+A~IHW"
        "t1OLqo0i3l#n8mWqli0vXULhztFq=9<83PnYK=K?0AxJRL*7yx(8APe*S~Cev(3^~|Nft*Ec%z47fG4Fe6g;}oK?KVkxcdd1h1;6"
        ">(WFw<KN#Ji?wAP>v0WV3t+7&L6)i;o1(H3#dJtWQ-^`~B3h@JK-x5Pm9EQ}XB&~{JWP1Y!??^c4zu)}TT$_}eydtt(@6elu?edZ"
        "M7z@y{g*EoJgY9vl1Pt<Hg(xm%}n-{Rh#DNucDUCQO=8WEt*z!cNK6XB1tyN1h0ui4UI@4z+Jq2k%%?JJw-{nfw?}4IJlA&WD}d~"
        "6AGIxzx?idKakxIX}hH}kds1`q3_izF94*mrmY3f6(xKSy79Iv?pRu|6aeaHJYOaJN(~L+e~Eu_e)%msVfarlV=K=K3=#uVq1>HM"
        "1QkDk<*qLDfUS5`TQYVUR~-mb2>+gEozSGyHA|bcXj)#xs%s`u6MiHhm^`ReXjGVp7Vzx{ArRJLaZ*dtxOGtZbSHLK16LrGJRm_q"
        "`wWGYqp;*!_}2Kr=L4%`Y2#XD;aI?;y*wm`uF8aiD7E`fQMQkUnugOLg=*GqG#Dubd$KlTn5%bSuNCU=x+?&<x+gAkKw`Q|v$Wl*"
        "wiDggFA)KgU_R%bsfLj@p~TPqg@+oVHeHt`Dghx~Ef@e;n=-o>`p|JqnBPGcC_@dlubF?WQGDVN&Lk@D=v56~rv=!^xg>nX*51I}"
        "=_lIW;+suYu6PzTt*E|PUY&oReD?Z7qWK|F20)lc!_!8v_n^+#)OlUkW$mxMw}T^*v;GVxvhLulH>|uD^+OGZcEvss;m=;s3wk34"
        "HhnLIt+gr7PCfQYT3J9o=$ScQ2|+c%1fGL61#3HJp9V$?XuV?9NPob3g45V^W!Go|+Hl~68LJRigpCjzl4ziBP5=>Aa6td<QaB|r"
        "jy#clN_BcW?hn&vXoP&Z<1^-Q=|Y>(a0Xt`Z)%s7Jh6A=^)U4eh3*?eI@SoyfFA@iV_t9DX@qxW;aiSCOBB_`20Eox+K3k4<rLo{"
        "SMAzuS*X#{dE#)#$hb?mMHL%t(h<)xKR|gJBhn`7;{)gv89;8aoLM54!%@(_lGcp%!j%z6Etd$ZR^ts$f};~130A5rN_DKl=57pq"
        "(K1|EjZ16qrlmmA4$Co|4(>EUz%@u3yt7CQ=mhrGL0A(;*?i6pnVU<`P+~2Hg*S5Wjkg5R?DP5t)`&MXN6B`8wjLj1nB@f_>3-CT"
        "$JWP|FzNED@ii0Rgdip@+W94TV^KiJ2T+}RUAsR0Q-5x}z7w@e!Og%zGk`q~<wni+-Fi23KxQ`Ch;<0Vc5koWy!+P^dnNvl*(sc@"
        "G;+Q552dwdzf)!sb|bCxz7Z2^{SWks(omary5%*hEfY=5D+tfQ@jwt!ix^aE7g{D7)<OyTw6N`P(WvQW?R9}Opz#Hc*MRk5pNqh0"
        "z>YL9`TygFZC1uOq_ah%)A+t}UN#Oe(v&cm9CR-~@*ZU5zXFwmO#pc)pald4O&<Wmi=A&MgrPHvc~TjiB?&C>eQ9ZqCMU^ELxMdZ"
        "OCbp=@B}m17Nhta`XEWfNG(k$&tq$_^&$xeZI>I|Bj>Ssd0NyUoPH8O-q!}7c09q$p{?18g1J#;hV!(c#QBPmal`~zCS5vMVE308"
        "8c~{je<3p*N|$SKpF&vDXWF4REH>T8V&q_!v=y`I*b@?ukoKoJ7?3m6s<XDd69wwzS+AWyxLBu~esiH=t-j6^O-^Rwkm!t_>ybtY"
        "-NF42yQIm0(FFERzW7X?-DKZ((5XVrW1~n%@MtLU#P>_B)Tcg<7M4d_dp&Vro|tXG=C9WvKyMOz$`A`*dfju_qnuY!R?3Ra+=I9j"
        "+8#u@*|trD^&vAY`D(miyqBm`1`ZXkMjYcU7oG_PZ)t&TFOW>ahaqP}%MCz7bDxCb&$#j!h5A$&RdeF5CI>I!*q@ZzlWpAb$wVPH"
        "q+|h7pgm!`qGj~ha}mTech)ZpSpGGwED_E2Y@CuD$Uw|P|1vQQD2>YwON%3z5&}&KjItbc?xfAM70@gguEKU|hJana%YJV;)<YU6"
        "WtXD2<i$q#X7CXN=+pqQ?3}%x>vdjBS<Bf6HT3TV`9VW1K>Fw{Yh{Qe#n6rqCz*$BB&@Du4Fry!S~~Jief`qMXiwV$cLqBH$il%r"
        "l=a=*BLk6uV~d3}4OF*7hb?N`3t;df+rvAt11m{F$5z7>KF6#p?y$@?!H&kcVXTK$bQNs07-pzck5XJ>7TbV23ZG&p`M!lfTCpw%"
        "S0S~Hi8;BqiYPSAh$hI%d^R^ABL_0kMM+*{!t-<Yh))$_<vcRCDbH5!GZ+Q#<Q^iU^x!hdft9jbm4!il`x1m6u1X7yE-my!jRM4S"
        "P>a61uR1Jkh@nP02FcSJiVqU55m}vygbqUP?Jw`%D6~tXrEu3+z%!O=CY2Buf1`8@h7OsQz<MW6Mpv1Y4WycsUO`kxRs;<4$&9TK"
        "Lo!!V{ONu5QRiHRE82=0xmt*e`288SJj2p81c^B7JUpMVbJbUyLFbY9gbmf?Vx(;aYIshX{lH}21y||?KI~f%Kj<ke<N{Y}p;rdI"
        "We~QduLLqBU3P8NwOTSh&2tRnO(TG?93A2rVA6vg;ifFabUYvtJ0ZZyd?}u^fdm>f(l=-XuFVbzyajYY%&stagW)++z)QU36v!~7"
        "VZly65}0#_3Ax-;nDLd!8dQVRi}Uj!gxtgr98>A+*{iv^*P*_OCIy1q6S;zGHbLgq+#EoagUp1D0YhJ-b5L@WoFsJ6t^G}lp{XDh"
        "!pkL?$r{MV>U$WQC>p(h^j(RETcT<u?(|FzgPePq8p$>qq&hX!pv}VWXY>B6jmq^pgWwre)lmwe9qa0bxPSKTL(U(`;Wu+QL)9j#"
        "GTuUidL5A8H}N>`@-CCX`?spmy2zVfiqr2@#{;i(2!-V`x-rhu3PScepI?Oi!eB8XaWCrK{8Epjn)tX)h-7~8N4;6fiX#kp=cqd~"
        "ZKKK!bTwuhhG2b2Zg%uhpp2DuSAP$u?rTAJ@Mt6t8%%y9vJX^_!QK$gdo7)S=<gbowFOy4jewjuQ5JAgcyjatc#SAM2Xyx*Hd6;>"
        "8*1}tW`;y(h9m>_nq3T3!JJxWhjHlnF$EhPg=*<x;MKyvjTv{6?%@C(@d8Y#qN4#o%VCR&6X5Xt;MzBU0U%e1ylQuHfit8G-k+)1"
        "7#662n=?sw&|1f>nZ>OEC?ZE!X!=r%S}=E_R=h2yTcxKjyGAvjTeED+c!q(~4l5=t>;yk{Xwx;1@j@)#mL<?D!~?ti<tDm$^Whq$"
        "b$k8uFCVU>_qRWOaMXOBb?s->B^I(GVbmg?s6s3);SGZiA7AMzEeybM<ogHH{fprea-_@XR#BY|c7}9Ohj7#qzDaogPwVb2tLK(#"
        "=o&?oh?wu@6vhC50({y0J<l2~QM`LH7VIkz&aWnP@HJvQIF@0ZLm$T)N!RoU_WcMxCnx)^od~)kKIxXK+JQ?K75?Fm_c5M7Db`)>"
        "FPo^+ib3MDv5jYf<kIUMB?jAb(5c7PpMpG4(Y#R$U{)%R{+f0f1a{VOOaJ)jpdcXHcHp~(%IA&x!xepXK^1)acJLg-0yOICZ&`1+"
        "(&Y$UZ;+ka6t7EA{EOvK2{eqCSpZXIkAqh%`Z42+2?9s$#&V;cLl7F|H=Z=Pg*kM$I+$(Cv73RkA%ngGVGVZNz}YMr_i@(5gV%Bs"
        "9{a`7<H)-B;k!Rue8*gN8WHfsh>~GfhC1VodH96Ap~Pj^hipK}EA`WhLF}X^VY{mY)nhUJH0!QL(h6iCZLI{qi*k_zZYh0j>^v$f"
        "H}w6(@NI`iISDIMa(&q17$?K8hlrLuwHhRq3cF>@G^!%m)jakWS%<TWe&ft)L>Us43wyvoSdRg^ljZfZ0!w}8TtTh>lm*hy)Dl7V"
        "YBMsJrqLFi>-S>mqe9s!s$+s*f@T`mWkgY`Fp-E4#v0T=B=cqn>!_!F(P`}wF{o7<6HoT^#dwjc$+A{S;IQCcNe<OwBPm<L33*kC"
        "BJn46i3J_@0?ojqykVt3>5D~4b{SaJB1keLowQ?S{$%1#L}gtBBR`WX`{JNarX~F+OI<gP6dIQpf|>`|Liu3T9fnFF{HN+bXh7OB"
        "!k~YAQIHE>P$A3Aoix1UImJ~s+1QA8T@~sl>@D;hQRI0}f#)lVJBJH9eP-q?>`aR~wCQ6CJ0@T3g!z)xnF$OCqgFh=gX}Bu$C&Sr"
        ">&dc^ld??>zyFp*R4Qx+-aJW#@_$u`uS0fYq&<|?P_J=D%L(iUaw?a_;;ZY_HbF>Vs7bBb;A;I`cX8N0!4GMJ-$0%f)06D!SjS0;"
        "v`!9+BJA2UYeLGgaRMdUk(Ju?`S<=b;Q%ag;slR}v8Wj*UM=S7%WvS`R*ZEG`7eIwF{io*mWi0}bo#H@MF0pf>j#vmS_K7)T`o-9"
        "a8#~HoRs>O-!xm+4W>ggsbk?~%6D58mXm&W!@N<>XbR@g7xTl%3^Qj)<TR?-<$t-kMUCrmIJ0f)DCwA-l@%w|<-;`n^p5*hl7{MR"
        "$qsOw8Ed4fw7v0}jG`~wMwMD=amGWfBcKAV`*g?iw9rWLU>>Q$$mqFq-6NuHe9UX2T@i2V5<kkBIx!IZ!jmzaEIJZM9Z?)RwL(3M"
        "Aqh`+wxt#ctFrP(;u;66$cl|=Vut>5#pyPnes2^xQ7q>pCaCwwK(%KFClM2SZ_2RPZkEQUmt}Kj<j|8pLhvT*(@l*^%CF`w+Rp|M"
        "(V#R992kC|gCB3CE6L>mF3hr29@mf4C<9Y`d2;fH^NA(U6^o?~b78UZ)wE6l)sJ?#=SX;+zv7YW(C>99qc)RiTnD^__wQo#5sbbh"
        "7}3$qBL??FaHy@jAZR7qM+4Ew-nG1@PG<ivRmP92s-vHM)g9~n_Vi(*zNhO@323^@V=6xS8HsAI<V@{w>)7;cMy0Cbt0k)Rd!)S?"
        "RQR_vGj79hta&-`nUG5k2F#xi@h9PZ9JH03#zD|b;&KGRW6-u2Sx(-+9ANtLc%bUx7U*X$lYmHFHa>u@?QQ$jK6h>3OpZek-91GL"
        "P@FdIn?4nR{X>}P^v~wcSMZQJoZ`Lx@xxC)Tz|ZN=Y9FFW}b`!jbi!O7g?i3mPC+Bl5i|>SijL3kZbMP+^lDmO=;2EH(CN`Vb!>g"
        "CK4kryxO&*v40bF$O}L1`R&xhy*!p}>-#5H*Ta6Z#n)Xy0W=)<3jF^Te_#;RT>%_Wmg+u_=(wL`eZ>XVvo5a={_sf&$B}_fYeZua"
        "L0<4bP)h>@6aWAK2mk;8Apo>CALbJX003qj001KZ0047xV=rQ9W^ZF*Y-eR{ZDB8Ta$#w1X>MmPaBpmBV|gxcd9@i$Z{s%h9w7gL"
        "RWH)iqiI^ShwS(;i%o$6(l%&kb`ORjFczJ#BT*_zC6meezwdjbNRhN`r$ASiSmOKs=1Z1kSD*WaZ@4B~+VbX!^c}oD5h`j@GtFen"
        "1y`DHNY`uL@-Ix1JJ}7rBHK<9(P?(ub)U%ZKYb{(EIU6tKjUrRNli9g|Kwj<s_%m{-O0^;DFmU42!RJ%*|p@BZ@Ud`?pWK>QcDV~"
        "#a-F5wv$g)FS}btMglkZa^3y*?wSim<pQ{gEx#Kivv8HeM%w3RHQRzVTRJq_pCI6tH%#Tl<@p(bXVbN9C&-dK+jo4j2PPHoM0QCs"
        "sx=cDUaF>(w6d0}9xk#)057Td1i%+Rh97Cu-&6Sd{deKltz_&As~$U9D>(ch2R3uuH*Dq*=qdc3{n2ruodi4k$7zPDjGxe1qldX5"
        "d8VG)mTAd1*#g9y-kdG)8T|}I1Z~+RdCnRMEWcx~e*mOMD%+v2lx96j|HIA(1VCC2Ol-QENu*yh`?J_h#QE0Avs^|J1OiM#tiXi)"
        "PPAbEY>g5f7iA*hOtG@e*0Fc-dG!h-wX7NI@kef>pjgCEnZ*a{_D6bbqN<vNygZ;?if7dWaQh=TPJFp?R;)S!awLkJR$@{hCKQ6~"
        "EAbO;R1|k!-e>RClYy~|qf;_HNZ}QmGV}pB*Hp70MNNw3_RoT}z^_^U(h26m)ERj!Tjd?oxzn_;AXA`0vBJ=NJ{bIq#5TJi7C|J%"
        "$iHbbuq!D$nP(rxC(%6$;-q~&mQKs94ETq_xq4ENua-wj3V~b-(QLk3B%AU_W&0JxCkBpq76j|1tM3mNge#(k9?L1K4@n!9r~*;Y"
        "h**qL0fTB4!5tOzKS1}*&GnVHArXw#icq3hPbI_xsNuq*cKiZa^;6ICPz+Ok3oij0zze#vr)%<-TpSh47z$xYJ19+{?NA`0xaJR7"
        "+%&smGL#~ph#97e3#F;p#G=_IP%KCeBo<`bbW|6`glgNVX_X~cTd;?A1Xd&RItEV;$=fCQ)&dl#^)pCbLCemxeW$6{kbUmi2JOEj"
        "`IPp%q=ZMALm1Qz$h4Uzuq()aLyJ8Jr9Gt?<ff&lz>uR<llk#uXy~y<Eui@oCS$w;NzA78&Hx9Z%}eqU3*Xo*vy!~=97x?cA8+o{"
        "LgCuy0u|Db3<|52^YN(Ex&$}1)69ZT1M`n9y5J#)Yc+(rTr84rQGE^q3M`pKFA|P|bv{NLX$~4`0x;>{d$!qT?}bTea{Zgp*-RI^"
        "G7-gOwiPwa=26$Te}&hGGs6#fnZDwfV0*cYL^%}X33o@R9D_b2h+;>FIDP&BL5X#Fp7G(}`j-{rH45SCxnzG1Trw=3x6mmX7>h@S"
        "p<N9<(|#ibAuOOv$lM@6s!myJdMyeVE1@8tO3hj2v~hMZjUvgSB$LuaN|R|`=bw#>C|nyL;x?cGgIr=N$fo)3i%($c<v}kT@_0~z"
        "usEjm-QzFdB2&JvvDNCQ)B61G$U3=gjZ(ysAiij%nQ@>#o|ewul}^j{P8|}Ck_IRUsHB!wBBUGSg3Fj4XpGTac-e1GjF%04qXiTU"
        "w+x(k=(ZQLSY|j>K@?*vU&b>)P;&uiXw0R>#)f^5ODvc@u&aewq_@!2vKFjVaHYl3X1Uj|-z0-cG#?vXC!5|5ys0N6*>><Zm=t^`"
        "ISX!itM%!N9FEOX*Bs}1#h9=S$hGLN4VAjWQLe>hf`VIkJ>bFt#Dnqko=S+N8+S~)TZ~t=0LGhUsL^V`2kMzB(;n$*qL9iAvDO-8"
        "$}=w#Phyt%<I+`5Fe$AQ4?>(l;~ZEV2zBnU3&R~0Kb=%|#-+CqYLmi`RDlPEmPs?<>PNm|QNAPXlGb$|aZS25d@Ug~rB;tz-+PY*"
        ")eUHlvz?8%bn;ox-MYy#bn^rN-Q_S-$H|;CQQOlzE6oMx9fpqm`OmCK6UK9OA$N-B7>|DoPZ!%EPn9<0nF-^g@VBVI#okw!<Gk^Y"
        "sc89D#ifd?`<RR?Bh<R2eGl5?c@pu?{eC~QTF2S(!o0oQiz0VLPKVC@NykZwSn2V60(=yH44>xVDNFa}l-b-v`Hr!k@y}J;LpPl~"
        "V{~#}iBoNyx|e(UTVCmA%=sr4;d$Q}8KP&GwsOrJ8!O<hWMwDR3Y?8@h%{Xv#H{e#pbHxU3sSud-K=e#-Ya~a1;dH5dLeAG{h1J="
        "gA_H`*oF5xXecmCK4a}GjB5SVro#en))sLkKfb&9HC<+p7qd<Gz;LQ*=2Cfz(XmBa+^I(={kGDGpO2fpq)Cq?R>4JhZZ+sOgDV?D"
        "soetCj9~7=9%SN}5&WoEvPpGx<<RDRiOs6X18<Ck#u#4jElz8^$r*gxq@NYsH2TtfK1^n}N5@<(s;%t4Fx<ww6pc1idkWlKUmZlj"
        "Rn@+gfKH{Kz=^miwF64^{zC<G-+d>uX+O4dYsPhYO)!g!%q_68{gp00sJCrBe2ODHSWn^P|Knn(y{e^xZ<*3#M7=zc#`1GCkm5?x"
        "aK~m9o$R~!{}zCM(O5%oM&2>=%rylw5nZ5iZ^;EgM$U3rW8gK}Bpj<22Hfq;L?;BhqtUL+@Js&h;zjpj*y1$~@d0bU#hEouU=T_D"
        "eU^*-qWpG2awFR}<f4Fkg<qU(9ZeIJ0GKo}TgWGl-h1Sg=MtJmyXnNmQGknRQAir_^Rxc}P)h>@6aWAK2mk;8ApmTEmJ5ji004;s"
        "001Tc0047xV=rQ9W^ZF*Y-eR{ZDB8Ta$#w1X>MmPaB^>EX>4h3XD)DgZIVH2+&~P4?*{(|;Zp(|yU?2tg&dL|N|&-ta}CN^;~g?H"
        "ay4={+h1SJrkj!ypM-6__w-3!*WG>iI6@*UtIz~{l<9y%un7AQsfR7Y{qFiJP|=W32|@uCvT6b>fd6`h${Mxrx~^^7#tSES44xgv"
        "nNq^uarR+`kQNq&o9BfjB+B}IA@|<@#i92cr*p@xnVQj-bw@ZIZ$wzCQ`=1B&w(H4RrjKp@c5t61}uisvpy(4Q7$8dtQ%02nclp+"
        "gsV?51*c=%fIT+W?OOduI8d1PS_WPWN%&1NpwsXRHE=FNq3Ex^+z(E;6f3G@oCc$sAR4ypj(rVXMR#<X>`Z-QQ;-YAbSzTM0+I5p"
        "d*hr9cZ^0ySzsZ2Do|)aF~xX<O7BIIyr-JRR%jm+DM{(!?!fT*+YfNKpbHwv6m83y)s|S+Wb0FgM{8n^H1)4DzZ8O@(7Ow`s0Y+-"
        "pxfffH1R~S@}pT!K<d2+{aiym5pUUz)r#@&zCxb+?y_zE08mQ<1QY-O00;m803iVA%C(=s2mk;t9RL6;0001Ua$_%IX=ZO@VQgn*"
        "ZEayMbaG*7ZfS03FLGsYZ*ysMZ*qBGb7pidaCxO!-EZSI5`SM{{|8q0BDGpy9JG7a1$+;8$O8A!V$ltX0zN}vWXdL@uOlhPZjk@I"
        "GyD=MTh4B)heRSd!<pfazZud=lKjH@PIA?W6Jf``6N;$rk+uAHs(9BD+HOfhRk>%|cYRm$@<cv=yeCyB2(4=(`H_@eTk+i>w18YB"
        "N%Hp1+c%Zy8q!m>ulWPvjn@1LK7HQ<uey@fJJvLGVRX20cfWnQ`{lns-Ti!DeExLzpU*QAO3+eCCv<PjW?S%xa&dR0O<%K1qFbuy"
        "qN}Q!x2%wravIXNZwNfS00n{-Y|9mIcZFbOC$=&-y|S;ln(I^ap=Lz|Jxuh1w+cG4Srfz<dTW7TbX&-75G5-__atXUqcyU^bqB%+"
        "CV0i!wy-0ZxTFIj$6K)SQWPq1FKE6hZV_Q69S~Sy+KZ;!vN|VQR?(qWg%c}!(N(-=60z|14Rkxqf!97JHnw(AAR0Nuo|QRaZ3%h>"
        ">a{Q5`o^N>4Ogr0Z+?95`)%7lmp|UZYJ4#Ic!eq#JEqb^bTvyd^394Q+Qh_;(eqL+8Tpmg1G^KV6KTQ@@Sy|mR=+^Bv|ZpJu2GU1"
        "EeKS0RAhw{t`qy2Y)r4Sn{OqIPy;|VwQURYKeZA$p<4h4gn-l9EU#SMi-BbsdV?KSLJ(Gi(P`#g4LbwpU>%wsElbBXe1H_xAX?J3"
        "^@&h2v>H-m%d3i1Lk(*O=-3gpXN0#>iJ{au9HeV?c39IF|7(+YhVM);0%;cze5~Q<#U*S1oOxgbm;p0eqm1H)NlAAMgTH<@NccdK"
        "K{Ao##4J{;pvg@BvhfsA0rRaK#1^#g>r+wgDS&_h?@+~pn7T=FkMq?>3eK4&-LaM#@+{0OdQ}n9noZQ6w~yeD5iv?_fWX=MtBrsN"
        "1fEhUInWmP7`dxgM^OR$qv(<Tlg-N}AC*n~7&|UIG#z!(wLJ38ggDw4&QH59%!rHILhH+dC|%Qo{R;s&;eDFDu8vn;<N5=sS(}cg"
        "W*cOY@%t)~*kLxx58cpisW>^>FwJ1IpKLM}`LOW@0r`XhgXBDZt#7nZx7do~tYY{wZ|iqUJz#nB{;IhFGe_aEWS@-TK|=#x%gDcC"
        "dd9hNjPoob@5mAcMST6Gwip{a$a_`YJvl3cVTj8$LF&1OwvcG~HreDM5$8elavP;=%vG*F2#v7B?<xqWb9}h~Da)>?%g9AZp`Nm>"
        "-{x^52ojj-IL~5EA5LnUZ*OjH{P@Gz=&WFDyITGuAM>9pTp-=oiV55tGj&fuA=C0CMMq)Fn5I>Nl07XSD`XQvpR66r8+@YOZ7~Za"
        "WGZYD*0UAr)MGAQ0m}#R)f7}5mWeJr1<<;4AiIi_`#qOL3@z{x2>gHG0{Bl%D_)2K8q@G2WRaTfD3ml-{lWq2Z{(w3>~{u92hv;3"
        "cN_=|Qpp|*A6kk|N?#D`vPY5!2F5b76W!2*VqG|5U>8o~zo}$cW=XQA2L_xAbr$-EiA&Okkpq`@8dE~_^&lM{EH0(hrXdB3`22z}"
        "H8`jUusaM%0nJPRz*rb9tbknmLYoYD*MWfOYIVV2KgIf7W&_jC)Slzaxi0?G$sTZ=8f@9wS%Al!YY#SXmQGd^digHiGw}Vx^46w%"
        "bCk*5OEJQK4J3fk(<uW{);$F{3u2+fX%ZP6UcyO<L*0ip$Jqg9GCpL3*@}1iU<G`+$4@=eZ&Bcfxk^yijc2W<c>Eb~{-x?{w6mG<"
        "F|$QqGZHc7#fv#_a*T=#s<XJsLIBGq)^0H=1OwQyWKe95WCCszhA)$~1WuGt`>x>dCsIA&@-z-<Ya@W0X40d1(DdfyCIi?mBd#7L"
        "0{+KS&H-n?bS#ai+w#3m1^6eA5=ii#2PwqAMeOYlUhq;h6zXp#)fE=d#}6-bjR^eN4qigdfhSid@$#>+&E!xNEWAECY540+kVT%C"
        "P^Bqpc~&oit1vW!jO?M;1F0Y#mnmdlW=atmh&fS9JX*s9Z0FPy>T*cFys5sLa5QXYNe7>xxViMW#h4%AF!8F6uON>=?;L@gpOs(b"
        "4t%m#Ddme<1D{u!10MspfHX|QELY&;G3fx|0CHY@`JvbAs5>~soeW?69r=b<3M&v4wgn{CHLg|)_U-eiH34h`f8n7sK%gyc3Ou}H"
        "ffs~6F`Q<P7a#__-?hAigGZo1msiEUtGC^tB5_+$8v6<pH&A2QHXP}Q*pzc;5)+=jA_)fu7{3;i)5TfU8Xlz+bAp*pYF?sn0BJ*q"
        "=I)?5+7r+}w?~p_8^FN7vMrL4T1`0s`ktLB#t2W}xu4oh>N{RSg)U%P)8snJ-pY~naS_B-NA-wN^(clKaWmPxF!$k$M3<3hWSc$m"
        "^a75VEvi6;@ifopTldjiqdacTyKwqPogE^e<Zi0*4WN#vfUea|E$!vLQ-$2q+aKOzlZJmXA&!^x@-(hy$NqC9VlcuhY=qp|z4jz!"
        "tj}Fsr%}Xna$H15?4v6Pitw;~T-fUneJ$Wui7VE~nK<gcXer}y;k}}D_9f=2brQO}(TsfNup1m-Foej<BQWJ&{djYLSL@+5O6B5c"
        "F3Q8Q&Y$J&8<1N+xE<(sR(yrf&|@JL>m{DJ-L*pBm3wHgpR_E8W~lY$-0J(>`67n~ycJ!uvUR^p2Ni92eToY(Pb7H%8r<swgJ?fI"
        "g}~Vx)lF-wM4p<4DZ$HRl+oOyI4c}T+$2xB^wp9H9VO$k%>E9sn*XUG&DO@3jVt6X{Z^6MUo1?UulbrD0`|Zm8=WROjf!#)04xCv"
        "`Z~3Rfd@8O<Y``G#~u=9V{Tdl_ggsghvSUFn6mUs#N|}c$tImabF{yGR9sFqFY&gp(4PDb6y78UXq_T9MlC+N5;yP-Pl!=egZ2@`"
        "{w=&XHgGoo!fWRc*Wt(1%+dQ*<ok3V1n=X$^dC@50|XQR000O8001EX+W}19Bmw{c9|QmZBme*ab8=%ZVrgb?V_|G(Wo>O?FLZKY"
        "X>Mt5XD@VeVQFq<axQRrg;Pyy+dvS#8~h&zof1%G)20_eN=ssw9AZLZ=uwP!XYHl?q3%aWLVtW`W!Wypp*l#b-M2II-n^-*YCZqh"
        "g4BUTf~VygWYEtPpf@OJ=Z&P$z!MQjFXC+d`0Z|WxBA{#*;G|EolGYxcncB}3WG6H1Z{gC(rIgi^q6)=Zw|)~{Cm8<(aH-nTe21#"
        ";hoak18(usQ_z-Ndg@7>!$*NiaiNDBOTv@`t$cbUl&^~rd?=Zwlds2`E;!5^x$AYvbUB?2dxibYkh`5ufFDB`oOatTV4@b3@tEK@"
        "th^(*9zJ`V)gt(AgGubVsNWc>-C5u^pw!HyLm{%w29x%#i;4R9?j4IS|3UQJIvqJu3o(VW2vUiI6dK#tcTtH@>?m}Ber2|UY+>WQ"
        "VaO6qq`iWY6mT5vy1iq%sM5v$dgS)B|Fo+fT+mr`T2V~x(DfSj9d1v8*Wh#Nb1G5W(HfWV;mc6n*bTPZGMC1-%-Ab!r~o3FV%(u7"
        "-0)tH0WBp8u^itl%A0A4pS@@OnnI#PyPbcU-OjH7orL@O)$D%$8ElRTHUy#Ah8M~Q0^cWXxv61Rr~rFFvLzZ=?|39gVK!+LnX3jz"
        "Wp)f^$2lY5LDU9T*%;WpvO0fNI7$tcFfubaGAjv=r1uNv7wN3ENBIXd&vI2m^`pcr<<{WXW6zwxOENV)ESBr~L}_4u5{Ju)*-|s6"
        "<hl3*P)h>@6aWAK2mk;8ApjsPk_V*@007%6000L7002x$Lq$$gMcrCkbKAxhe$Pz)!wzTKk!B#;ag(%3o@7go**cVml<Udk5?GL1"
        "16X7i5XJubp3B}qQnEXJtH+aA5ccfZbNSA9&O(SU{JHLBxl`i3E>-8$x8FST>OYlr+H~U8{AD5jF1u0MgLw7w<@Zmo+6_bh^NSbz"
        "{eCWKqq(u$7Y%RbUVQrvExo^8{k9g@%Nz0LZh5n~U)(L%;_cl^d|claVs-mrb$9de4L&aDt((RAezkc05ubPw-_6C1s&%Ia(9q5O"
        "Q}{FUCz}bklT9O9B|9;|i3ep{CuCQN(sUKC6Sc8obgB?m_108liI0mQ3tz6Z8?4@p_zNMOsBl+RiOoT*RmoDmgAJ@1w>$BZFg5(3"
        ";XPB1t?GtT)r>vStu+0?>g{e2X5T3*Kr2|Phl7ygurpTwn|2I>&Mg^sauBfZ)=F5}ZRsU%i-`_uD;sf3(ogglJ3J*FK?zxsoS+EU"
        "0^SXB47}l$)XMShVAsK#rV!Gq@Iyn27kDCkI(8K-D^1&)E{O7PioG6oEQq(9i?_y-CjDr81M!oL8(klLHS=O-bQ&iv^d+w~du0o7"
        "qXloF8oJ|u7h*7?l;C}QFNopakrk|HWhb`^y@)%zvD|sp3bEfQvH*kuZ7fOn#5#M8fdyhNH0VNJay#9ls9M)x(q7pTWncX8@=uqv"
        "p8*rG^`oS5a0A#LJps<L$_0@ibE7&iM5)1{6Y-g*x$A#7<4jz@YW$a-UFP10zi5~TU5zNsiabt&1oa3SYlpgk?yYu?!k<EpQ-Zv6"
        "ChFF(VF@`2`PLo>U2j#btOWt|+nNmj88>N7rQv*%vMhR9cV#n@K_DwdX9m&eR-*#oP-p64kCEnRM{rIBP6)|Jl7bw5qX>{t>+NXi"
        "1#nNJCYW-!`4>dRsrs@z@H6ns*igFE*0k`;awj{`Jmw2TNayfX8G?(RH2%k0NWnHDdByaQLCDcrA&q*C31&z=f7UGo1)KvOPfk4n"
        "H8}1AgS$gPoXoANv=qakpPv1bv7b+XVh;~VYl0_?yOa{T3r9%_!3OkaYh?xK@*s61H%-8V46X{qS&YU~`mmA-D+20)E#U14du%Uw"
        "Ta$6+Fktl|GX;Hvj0-q}dX#Msd&5e=eTaHqiLYMwy@I_TA-fv0zsyZ>qpW@aYd$D}R&lfASio)08^T*4h+;zo<u}ry7duMM3OB)I"
        "fIwmh!@bCDm>T<?E_XRXfY%175Rj}^J!tYN#x&T+=Z#QcXJf-pAjA7R&#oX1>!5NFG~@&c+nI)P1lH-T?qHi!-=89Ngs*x6357T|"
        "p|>DLws#@P_Z7veR_c_ns+X2R25mzpX_ZyY0g|HoOvc+l*kHhPvQ?MC`5Ms4*0Q7mQshN6TJ}UOw2(4&?)^82^PUNu_55){#2n7I"
        "jArx+7^-bl1*J_~L-AaB#u7ytwh%3Y7oIvmk!8q$CBeYnO+<7zZU9a^o&<VAaYo9Mf<7;3CxZAB@)5mF_3+tJk`o$)YuX<}f1|*l"
        "HCW>rlJj+pK+NK)W<eNpf(X~JN;Qy<)&NKs=!%VOD4_Nh7j{HZ#?HGyU_wkRuadFQR)b54Lr!qTvt=s5S-uPWO<Dm;>IPRgKtMrU"
        "R?i}1atAl4)=i)tit?zixRg{v{VN;=SQnYoL{gSpzQ|!^5)7Hi(RRRz%h3@Jpe<X%P|rC(5e8;eOFafdOphA^6HeoLU5;jSkjkyJ"
        "pAiggB2kf>DW|s`)dYx1^b}e5Y)B${&6Z$vDe}CX&(0<E5hIUB59#=2W|W%|F|$3|SL{Haje@`edZ`HCLGOH{EF+!zV}w9zaO2W|"
        ")w!G_KhAP7K+~(a_!S8l?))YiA#gCU9=Y`Tusz!a<w=)=wSpQdGAjwh6wsRpA2A0Yl5kQWgS{HSJTVdh4x4JPk&kqyyP^<qaBTeX"
        "3b>KoV$Wp`vKbCnwN>zw1`7CKO2qfmYVSKp+%j}(umm!$$Cy3=zYOO4aRZCNmJqakBO&VJ6HuG$u%joQdgMLdq&0{LP4xO?&okAY"
        "z$W<VFPWb{NW|j*4<}u~I@J%DaL}m?fk=UhjytJK(erULCj!9-X*>C#hz&-Ss24YNjf4=YgK7Yk`M&@%#t!VChzy=qd-_Po6Ay=G"
        "U{{5GW#2d0teOs-Po_pZ^U9Tt)L?XeH$NYklBDD&j%eCJHajP+rev)x08(g#l#X>VM}iA?34OHbl&_Y6i9lLn8$j!h7sNAhhwN)H"
        "oPf#TR6W1P9{myO;9M+f41?4{IY2Lr@93IAZ&?YsmH2A{KHvLaq*|7kxHZmQk=5`dr5TZB@y`$dQZ#b!Mm^wZ8@1)?09Fg?XPn^("
        "DxU*ADi*A|^Syc|s7x+8gp&oYw8Uva4)e8XNJd%}T3Mepp*2bQ=F4GVUR*UWU(ts#W~2+Wu7al_3ZscZ9JZKMfQIkqVx{sy)0}p1"
        "<sqTnF)#r(>cD@d5cLcjB+nqfgxy8}J_;ygJ@9{%D)zMR<{JM5x)rH^BCBLU)+)vBu1y1-6cc~|XFsPpd?7FSxFbZ}7Ij7yxkmv9"
        "Yd9wYS<aJFV~KxGPAjQ$95)}oaG8zUZnAB84wR@j_Fp&z;2Dv{aDtXYcNjC=dAeK!1il#IC=}cOE!jk|pySQUY;&syNVy^q?Q9M}"
        "9R(<Le8PP1xMvEE0&^}^&%#H0fpA$VWLQN`IVfU>lz{$#Jjytu(FxY#pTzPQR)Y{yuQEh{p#<Sc(Wsn*cF4*aALg{0ADgbaL^O-B"
        ";M+#@`fPc3zj$*ygM@q>$nKblej8-z`Br(pWx#MIbx!O{Zp&qb)>VQpWkoG&ia>SNxQJB}C&{@Sk9LF?d=@%I@io@X<(xP4S?f~R"
        "fsjT?hmF|eO2Ds42@DJhw*5IMDub5E;K@qU_;Sx`@k<W$lW?14`gEctbe+%#Yw9*t$kPpsEzX!)240xW&3s33#!~fh+7Qct_K{r%"
        "5^Z&b=R8CYbvW~bjvLvBlF-85@3?_OP&;Gm%<aT5xEGI;L+CtG?~6R<NFkqTgkp!u9Y8E8va0ak78|8JG;=XQdvB$$Ns5A9;J_P`"
        "V^J%HQ<ADuT{X6W<W3?dAPqN<!Ko)OM^+BA5HQ8r#G=_3G+WFS?06E0Y@Da>NX{B6^&-S_=~94M!qFrx_XtYyhso8(DW67?J7L9V"
        "NPjjiJabLOa<c0wQ=ieN$a1KrUi0uo2a^x{VqTFz+&iBTCKXPu$R=ev@^l<k6Zgh|nof9PgLBk99Qox3>XQ7O1NXX#GrGC>*g=(b"
        "<U92UdzBg+Xp)&-$D~Qg;h1!0)3IzU_T(rmYY%YG<Eao6!%aTj{;#?^&&Wv8EOJ-|Q{5`weX@Y18E{p+E~3h_G2G8#8gHpJ#QH+Y"
        "xe>~QQ<dVYC(Mc5Gkz;3AUt{nQ^sCu3;iG^!NF$>HFN4wl{xei9!H~CwUw5yvW{Dpc;)aTAY@=E4$(5>-jyLh9hjzPcTi}4f04%="
        "C*GLI7H4sh_uwE+*$2Ei@IQiDKA!pI5Wqp{7&58Nwd#*i`}-;^_zqlyC7)ai%)qpGQI7h6t$~@8a7gcv)U4nX_+<7Qax1vc*NHQ="
        "A(-p0b8({`wOx2`T#HY@R>3%jm=RIi&4D{$>fW(E%YcF0No`M>tQE;|9t~X5_5#(#(ZflPo4<~O(TSHYaZ(1wV0OJ0i}g&rzFsfZ"
        "(Zrt?_wVjL-iuGytJU@LesQ}NcdPsxfxEZjdilHf$6|Q{oJjK>^dk;lU3M5vSW{&q*pyE+gOvfh4$vl(xu})0XOI^xdcU}Te_Md@"
        "msgAB+tp(E>+Ns1%lkt7cDs7>4)nZ!y?DR4|D9s#?c#oUyXLPMu7i*d*DG+`$M@GO@!{j@!`=Fpt1;hGHh6;pN9bW&O}86#3&c(3"
        "QQUw7t?8{sK2L|PA<pm}io^uO*({c)n9c!l!V^ar*N!0HnNr72o6+81^U|m{zvn&ehC>woZ4OU@VR6}eEjPNMOZ)|vW&tFr8%PzF"
        "0*@M+qk@{yt><H#aCZigG~}a)PHh_vEUi?RMZ9z`CX?bcNBUyyUoe-zn}bGgi20FvTbyg83zD$ufS*G+x}`s#D~zd=x`G3q=($GI"
        ")_$-+E|#s_PHzfudH6aZeW&1hg?Ie<1vq3EuwcHjK$^o7W4r|PQUh?~7!NeZfw<+Hd#v)QmgBYcaidHo9V6<D{6u%&r5V^x<_;In"
        "Ud#u*@Z^o*(A^qS?R7JmNPdQzZTeo~>=9YRh`QHOH>2f3C7XKe5&@zDdG6a3yllWY&TYb5D+d9D0f~hEc(xaW#DTr69yDFg*8Y1E"
        "$OdnS@TrBDAF|=6xwtN|SfD8a9^>ZMsgP$0@@a=$f114~mz2+5@dd(K?u_C2DveO5SN=551%g*ALKuKQ(ouG$;)C=&VD*T2pa@iL"
        "ho8G-Gd4DE6Vw-G)A+F#ahn$iL`akQUI&hZx#Qb19bq*#CGX50`zP)~qtVE!ndEdt`p%|n@{5;9(fth!O^E$tMAw9^q%pCfbYYVL"
        "eHuGu@!%)U*lOvTQ5*B0)1T~9C;L>Yh8~BP0FSQD4Cke737P^GkCx1`d$cy)(D}hMAT>1g*y!?nwm6+iZw{V~WM?{{6_Q~i`QK+@"
        "neof0O^D^&<qcNdb6@QFH`1NctCz2Sgj}$r`8>#t__KJeYg5YRSJk#M*VOS!9|Tp#k*8lL7WpTL;sRfYpZ8t<LNWsTM4vJ!`Qd8d"
        "Yr6Bz2L*bZ52OMnQp0C9G&PZv;Y|QVIiwEpOW!%Z2b#8A-(G>*e$m&Ev8O7F|D=iJWaE|74`*<&oWJlS*ZLocR$y3cy|q#kCBlU<"
        "l`UvP;dI@01b_s*hg9#5KOYPIa-wGL^k}fve*jQR0Rj{N6aWAK2mk;8Apq&FOCQz&004*$000;O0000000000004ji00000V{dL|"
        "X=igUWq5F9a%pX4Zgg`lYIARHP)h*<6aW+e000O8001EXI@WR1U<3dFjSBz(9RL6T0000000000fB_5w003ieZf0p`b1!mbaBp*I"
        "bZ>HbUvqSIWO*)Xb8l`?O928D02BZK00;m803iTfDLbZS0RRB)0ssIS00000000000001h0jCB40Ap`%W@%@0FLZKYX>KlMcwudD"
        "Y-KKLb8l`?O928D02BZK00;m803iV9ixwt(0RRAr0ssIh00000000000001h0YwM^0A+Y#ZE$R5b1!mbaBp*IbZ>HbUvqSIWO*-T"
        "c42H^a&LEYE^2dcZfsCX0Rj{N6aWAK2mk;8ApqrL91X$%000{S001KZ0000000000004ji2MPcHWq4t2aBO9BFLGsYZ*ysMZ*qBG"
        "b98lNc`s5$K}1bOE^TB`O928D02BZK00;m803iT-E!Q^S0001s0RRAL00000000000001h0SXHM0A+Y#ZE$R5b1!mbaBp*IbZ>Hb"
        "UvqSIWO*-hZeeh9Xm50LFEL_fWMO7DHaIjkIWjO~HD+cwWi~TqGdVIdH8n6|F*q<aH8eRmWimKrHexenHa9jjVKQMfG&f^0H#uW9"
        "G%jj$Z*EXa0Rj{N6aWAK2mk;8AppFVcbeV+004^t003zK0000000000004jiaSQ+eWq4t2aBO9BFLGsYZ*ysMZ*qBGb98lNc`tKr"
        "VQ_P3Z*+4nF=b+8F=RPoWHB-`V=*%`HDxnmIW{#lGi7EpH83$^G-NO{G-hIDVmM(qIWuKuF*ac^W;tPIWj8f2H!)*1E^2dcZcs}B"
        "0u%rg000080000X0Ozjx=HLJT0E+<t0BHaK00000000000DuA34FCXTcwudDY-MvVa%FIDb7^#Ma(Q2KbaiBTFLQ2TaC2yHbaO8="
        "WHd2iGGsV0FlA+CH8W#lIALKmGG#erGh{S1H#BB7Vlyx_Wn?j7G%#f|I5RRgF=l0BWjSUsI5#*oGC4LbYIARHP)h*<6aW+e000O8"
        "001EXE-TfY;Q#;tiva)tX#fBK0000000000fB`)Z003opVQp}1WpgibWpHnEX>@OLd0%sMb!2%jb8ca9b7*gLb1yVBVPs@8F)}q}"
        "Vl*)_HaKQ6V>B{iHZVD4HZU<{Fl96~Ic7OHF)}t{He_L9GB_|aFgQ3dVqs!oF*#;3G%_w~b8l`?O928D02BZK00;m803iSth}c2j"
        "0001s0RRAL00000000000001h0i_TC0A+Y#ZE$R5b1!mbaBp*IbZ>HbUvqSIWO*-hZeeh9Xm50LFEuzeFk@n2Ff}nbW;J6qF*GqV"
        "HD+dJHDoYkGBjp2VmM<lH#9RiH(@wAHaTQ4Wnp79Ic8!xWiw<sF=RM2IWB5*Z*EXa0Rj{N6aWAK2mk;8Appz=2w~s=004>s003zK"
        "0000000000004ji3labTWq4t2aBO9BFLGsYZ*ysMZ*qBGb98lNc`tKrVQ_P3Z*+4nH8^5qWi(-7W@RxlIWS=|WMX7uFk@siI599b"
        "GG;k5WimKoV>dQAGc{&oW-&7|HfA?8F)=q{F*7z}Wn(xwE^2dcZcs}B0u%rg000080000X05d!75#Rs-0E+<t0BHaK0000000000"
        "0Du8<6952ZcwudDY-MvVa%FIDb7^#Ma(Q2KbaiBTFLQ2TaC2yHbaO8@HZd?YWjJMJGh;b2GC4M4GchtZVP!ZlGBz<YH)SzqIWuE2"
        "H)Sv|IAk+2WHmWBIAvjGWH(}EG-EI~Fl09_YIARHP)h*<6aW+e000O8001EXb&f<(-~a#siva)tX#fBK0000000000fC1SQ003op"
        "VQp}1WpgibWpHnEX>@OLd0%sMb!2%jb8ca9b7*gLb1z~!HZ){mIWc5oVKrklGdM9hF=b&iH#an7Vq;-uI5aXjVmDziHZ(OcFk&|`"
        "Ib~sEFk~?_H8x@~H)UjDVrDLCb8l`?O928D02BZK00;m803iSk$`O&^0001s0RRAL00000000000001h0YDZ20A+Y#ZE$R5b1!mb"
        "aBp*IbZ>HbUvqSIWO*-hZeeh9Xm50LFJ)w6IXN*mVl_2kWMpGBGB7b>WH~WoH!?FgFgZ9jH8^HvVlrl7I5jvhVP#`uH8V73H)LXA"
        "G%+?}H#jw9WG-rRZ*EXa0Rj{N6aWAK2mk;8Aplm-+vP(7000;d001fg0000000000004jirxySKWq4t2aBO9BFLGsYZ*ysMZ*qBG"
        "b98lNc`tNwVQFq(a&LEYE^2dcZfsCX0Rj{N6aWAK2mk;8ApiutO~YmY003+N000~S0000000000004jiG8zB?b8=%ZVrgb?V_|G("
        "Wo>O?FJE72ZfSI1UoLQYP)h*<6aW+e000O8001EX^CrSQlLY_(j1m9<8~^|S0000000000fC17P0047xV=rQ9W^ZF*Y-eR{ZDB88"
        "Uu|J&ZeL$6aCuNm0Rj{N6aWAK2mk;8AplujE6q;;003A3001HY0000000000004jipCAALb8=%ZVrgb?V_|G(Wo>O?FJWh8ZgekS"
        "UukY>bYEXCaCuNm0Rj{N6aWAK2mk;8Apm0VNhEUt005Z+0015U0000000000004jiE+GH_b8=%ZVrgb?V_|G(Wo>O?FJWh8Zgekf"
        "Z*OoeaCuNm0Rj{N6aWAK2mk;8ApixHO*KdW002q=001KZ0000000000004ji-68-0b8=%ZVrgb?V_|G(Wo>O?FJW_YWpr~dUtei%"
        "X>?y-E^v8JO928D02BZK00;m803iUDqh(Cv0ssJE3IG5n00000000000001h0cIlr0CRF<FJftCZ)0I>XJu_|VJ~5Gb7gdMFK1<K"
        "ZDBNDWinzeYIARHP)h*<6aW+e000O8001EXxU6!B;{pHxCkp@oCjbBd0000000000fB}vs0047xV=rQ9W^ZF*Y-eR{ZDB8Ab8}^M"
        "b1!FQZEayRUu866E^2dcZcs}B0u%rg000080000X0E6usgdGC_051vv051Rl00000000000Du9uDF6U-a$_%IX=ZO@VQgn*ZEayM"
        "VRLh3baO9sZ);_4X?kUHUu|J-X=Y_}bS`RhZ*EXa0Rj{N6aWAK2mk;8Appu<X{n9@000C7001EX0000000000004jiAT0m@b8=%Z"
        "Vrgb?V_|G(Wo>O?FJowBV{344X>N3BZf7oVc~DCM0u%rg000080000X0Efb+seA<h01XZR02%-Q00000000000DuAQE&u>?a$_%I"
        "X=ZO@VQgn*ZEayMV{dL|X=g5Qc~DCM0u%rg000080000X03zr9XD<T)0CNcd0384T00000000000Du9ZGyni|a$_%IX=ZO@VQgn*"
        "ZEayMV{dMBa$#e1b1rasP)h*<6aW+e000O8001EXRRug!g8%>kvH$=8ApigX0000000000fB_3Q0047xV=rQ9W^ZF*Y-eR{ZDB8D"
        "VRT_HUtei%X>?y-E^v8JO928D02BZK00;m803iTpza8qk0ssK11pojc00000000000001h0nIo70CRF<FJftCZ)0I>XJu_|VJ~E1"
        "bYU-HWMOc0WpZ;aaCuNm0Rj{N6aWAK2mk;8ApoY)O~KFs003zN000~S0000000000004ji$2$N3b8=%ZVrgb?V_|G(Wo>O?FJxhK"
        "VJ~5GbS`jtP)h*<6aW+e000O8001EX<JN{j4FdoG#|Z!cApigX0000000000fC14x0047xV=rQ9W^ZF*Y-eR{ZDB8DVRT_HV{dG1"
        "VRUbDE^v8JO928D02BZK00;m803iTANXHGx1^@sN5&!@o00000000000001h0UbdA0CRF<FJftCZ)0I>XJu_|VJ~E1bYU-JZ*FvD"
        "cyumsc~DCM0u%rg000080000X0QNeZm2w#X0902103iSX00000000000Du7_NB{tHa$_%IX=ZO@VQgn*ZEayMWMOn+FJpCbVRUJ4"
        "ZZ2?nP)h*<6aW+e000O8001EXe)w6WHV6O!zZ?JnBme*a0000000000fC1ED0047xV=rQ9W^ZF*Y-eR{ZDB8DVRT_HWNBe%Zf|pR"
        "X=8IPaCuNm0Rj{N6aWAK2mk;8AplD$=9j1y002cr001HY0000000000004jiOlklCb8=%ZVrgb?V_|G(Wo>O?FJxhKVJ~KNY;0d}"
        "Z*OcaaCuNm0Rj{N6aWAK2mk;8AppME8Bu=?000jz001Na0000000000004jiG=KmAb8=%ZVrgb?V_|G(Wo>O?FJxhKVJ~lHW^8G0"
        "WnXh<bS`jtP)h*<6aW+e000O8001EX!AbBzO9ub|_7eaAAOHXW0000000000fC2T50047xV=rQ9W^ZF*Y-eR{ZDB8DVRT_HaA9;~"
        "Xk~LQaCuNm0Rj{N6aWAK2mk;8ApnaCH|N^}0027+001Wd0000000000004jiew6?Kb8=%ZVrgb?V_|G(Wo>O?FJxhKVJ~obbZBpG"
        "UvYI|Y-x0PE^v8JO928D02BZK00;m803iT=t?Q7%4FCXpEdT%`00000000000001h0iKxv0CRF<FJftCZ)0I>XJu_|VJ~E1bYU-Y"
        "WpHnEX>@OLc`k5yP)h*<6aW+e000O8001EX$i!}aa~uEw#ApBjDgXcg0000000000fB~SV0047xV=rQ9W^ZF*Y-eR{ZDB8DVRT_H"
        "a%FIDb7^#Ma(Q24Z*p*Tb1rasP)h*<6aW+e000O8001EX{};W?_Xq$0rW^nOApigX0000000000fB{>?0047xV=rQ9W^ZF*Y-eR{"
        "ZDB8DVRT_Hb9ZHPZE199E^v8JO928D02BZK00;m803iU+WD>Kr2LJ%o6#xJw00000000000001h0g=rB0CRF<FJftCZ)0I>XJu_|"
        "VJ~E1bYU-aZ);_4X?kIFX>V>WaCuNm0Rj{N6aWAK2mk;8ApoTYH=F4I007$o001Wd0000000000004jiiPZoAb8=%ZVrgb?V_|G("
        "Wo>O?FJ*RNY;|FDX>V>XUtei%X>?y-E^v8JO928D02BZK00;m803iTe#NZua2LJ%{7XSb#00000000000001h0k_rw0CRF<FJftC"
        "Z)0I>XJu_|VJ~HNVQh6_bZKvHFJ*RNY+qw<WMwXJc~DCM0u%rg000080000X05_!f&KeW|0M|zV03-ka00000000000Du8r-T(k|"
        "a$_%IX=ZO@VQgn*ZEayMWp-g~bzyXAZ*DJVb8R?lE^v8JO928D02BZK00;m803iVU1ZD}Y0RRB{0ssIe00000000000001h0lM-4"
        "0CRF<FJftCZ)0I>XJu_|VJ~HNVQh6_bZKvHFKA(MZe??GE^v8JO928D02BZK00;m803iV8>g^7s3jhFcB>(^{00000000000001h"
        "0j%@@0CRF<FJftCZ)0I>XJu_|VJ~HNVQh6_bZKvHFK%IUX?A5_aCvlSZ*DGdc~DCM0u%rg000080000X0JanEsMi4i02TxQ04e|g"
        "00000000000Du9S0097Va$_%IX=ZO@VQgn*ZEayMWp-g~bzyXAZ*DJea%E&`V{~b6ZgVbhc~DCM0u%rg000080000X02JkHTtEN-"
        "07C!(044wc00000000000Du9x0s#PXa$_%IX=ZO@VQgn*ZEayMX>Mj^a%FB~WiMY}X>MtBUtcb8c~DCM0u%rg000080000X04kA3"
        "%HRh80OT7004M+e00000000000Du8K0|5YYa$_%IX=ZO@VQgn*ZEayMX>Mj^a%FB~WiMxCZe?;|bZKvHE^v8JO928D02BZK00;m8"
        "03iT%;)EM10RRBx0ssIb00000000000001h0b>gR0CRF<FJftCZ)0I>XJu_|VJ~fOWMyn~FJE72ZfSI1UoLQYP)h*<6aW+e000O8"
        "001EXIFgjua|ZwbJ{JH0ApigX0000000000fC0@60RVGyV=rQ9W^ZF*Y-eR{ZDB8MZ)9a`b1!0YX=G<*E^v8JO928D02BZK00;m8"
        "03iSgX`>3i0RR9)0{{RZ00000000000001h0e=($0CRF<FJftCZ)0I>XJu_|VJ~fOWMyn~FJob2Xk{*Nc~DCM0u%rg000080000X"
        "0LVsj7F7fQ0Hq5604@Lk00000000000Du8}76AZra$_%IX=ZO@VQgn*ZEayMZEs{{Y;!MTXk}w-aBpdDbYFCGVQzC~WpXZXc~DCM"
        "0u%rg000080000X0Qo1z8*l>v0D}nt051Rl00000000000Du7<8vy`wa$_%IX=ZO@VQgn*ZEayMZEs{{Y;!MTZ)9a(XL4b1XkT!0"
        "Z)#;@bZ>GlaCuNm0Rj{N6aWAK2mk;8Apq*<Bm2k#001Qi001Qb0000000000004ji+8zM_b8=%ZVrgb?V_|G(Wo>O?FKusRWo&aV"
        "Wo=?*WMpY>XLBxac~DCM0u%rg000080000X0JH>t$`S(r00IaA04V?f00000000000DuAIAprn$a$_%IX=ZO@VQgn*ZEayMZEs{{"
        "Y;!MZZfSIBVQgu7VRUJ4ZZ2?nP)h*<6aW+e000O8001EX#-x%R#|8iZixvO?D*ylh0000000000fB`-w0RVGyV=rQ9W^ZF*Y-eR{"
        "ZDB8MZ)9a`b1!XWa%W{^Utx4~Wo~q7Z*DGdc~DCM0u%rg000080000X03MT~bg=>e0D}eq03ZMW00000000000Du8bECB#>a$_%I"
        "X=ZO@VQgn*ZEayMZEs{{Y;!MdZ)9a`E^v8JO928D02BZK00;m803iTvMhxiy4*&q9GXMY~00000000000001h0X;7P0CRF<FJftC"
        "Z)0I>XJu_|VJ~fOWMyn~FK%IUX?A5UaCuNm0Rj{N6aWAK2mk;8Apomc;kJJV002!F001Qb0000000000004jid_DmHb8=%ZVrgb?"
        "V_|G(Wo>O?FKusRWo&aVaB^jIa$#w1Wn?aJc~DCM0u%rg000080000X02N1gdYc6R03;9q03-ka00000000000Du8LMgahGa$_%I"
        "X=ZO@VQgn*ZEayMZEs{{Y;!MiWoBh^Wo~0-E^v8JO928D02BZK00;m803iT_7_N^!1^@uN5C8xo00000000000001h0UJyK0CRF<"
        "FJftCZ)0I>XJu_|VJ~fOWMyn~FLG~nbZKs9E^v8JO928D02BZK00;m803iUz0#RGu6aWB7RR91a00000000000001h0hm$&0CRF<"
        "FJftCZ)0I>XJu_|VJ~uJaBp*IbZ>HbUvqSIWO*)dc~DCM0u%rg000080000X0JZv$Z{7s}0ND@#02}}S00000000000Du9wX8{0n"
        "a$_%IX=ZO@VQgn*ZEayMb7Ns<W^Zg{E^v8JO928D02BZK00;m803iU|PzfTA0002J0000b00000000000001h0nlv$0CRF<FJftC"
        "Z)0I>XJu_|VJ~!YVQFq@Zf7rFUukY>bYEXCaCuNm0Rj{N6aWAK2mk;8AplPI{P30r002J}001Ze0000000000004jip>6>Hb8=%Z"
        "Vrgb?V_|G(Wo>O?FLZKYX>Mt5XD?%Qa$$63WM6Y;bS`jtP)h*<6aW+e000O8001EXg{`{f9svLV<pBTyDF6Tf0000000000fB}7V"
        "0RVGyV=rQ9W^ZF*Y-eR{ZDB8Ta$#w1X>MmPWNCABX>4p^bZKvHE^v8JO928D02BZK00;m803iTuY=d(c4gdhNF8}~30000000000"
        "0001h0pfN60CRF<FJftCZ)0I>XJu_|VJ~!YVQFq@Zf7rTWo>VAd0%jHZ(?OGaCuNm0Rj{N6aWAK2mk;8Apja<yqj17005K%001Wd"
        "0000000000004jiL4^STb8=%ZVrgb?V_|G(Wo>O?FLZKYX>Mt5XD@GJYGq?|X?A6EE^v8JO928D02BZK00;m803iS^;~(v?5C8y|"
        "KL7wI00000000000001h0p5lI0CRF<FJftCZ)0I>XJu_|VJ~!YVQFq@Zf7rVW@c<@Ze?F{W^^uac~DCM0u%rg000080000X0JJtA"
        "<`W120A?Hj03!eZ00000000000DuA1mjM8Ca$_%IX=ZO@VQgn*ZEayMbaG*7ZfS03FK};cX=8aVaCuNm0Rj{N6aWAK2mk;8ApmTE"
        "mJ5ji004;s001Tc0000000000004jiD4+oVb8=%ZVrgb?V_|G(Wo>O?FLZKYX>Mt5XD@JaZ)Rz1X>MmOaCuNm0Rj{N6aWAK2mk;8"
        "Apq#gwV%KU001u?001ih0000000000004ji^`QX(b8=%ZVrgb?V_|G(Wo>O?FLZKYX>Mt5XD@PPaBp*IbZ>HbUvp-3E^v8JO928D"
        "02BZK00;m803iU|0ZiW{0ssIX1ONae00000000000001h0sg810CRF<FJftCZ)0I>XJu_|VJ~!YVQFq@Zf7raa$#w1WpXZXc~DCM"
        "0u%rg000080000X03a=r2c-@G0NW}600#g700000000000Du8xtpNZ`Nkc_WQ$<iq1qJ{B002t>O8`_F001t$0RR91"
    )
    bundle_sha = "8d1aed91bdb0ff2179f9380e3d89b3d0d1429dbf13a86439253cf734fe6461f2"
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
    PREPARE_FULL_POOL,
    RUN_TRAINING,
    TRAINING_DATA_MODE,
    CURATED_POOL_DIR,
    FOUNDATION_POOL_DIR,
    CORPUS_KIND,
    REPOSITORY_SOURCE,
    REPOSITORY_TARGET_COUNT,
    REPOSITORY_PER_REPO_LIMIT,
    REPOSITORY_DISK_GIB,
    TOKENIZER_CACHE,
    CONTEXT_LIMIT,
    TRAIN_DECODER_LIMIT,
    Path,
    runtime_root,
    get_experiment,
    fetch_tokenizer_assets,
    hf_token,
    json,
    mo,
):
    from bifocalgemma.data.full_pool import prepare_full_corpus
    from bifocalgemma.data.offline_sft import load_curated_pool, validate_curated_pool_limits
    from bifocalgemma.data.repository_corpus import prepare_repository_corpus
    from bifocalgemma.training.curated_sft import fit_curated_sft, prepare_curated_training

    full_pool_status = {"status": "DISABLED"}
    if PREPARE_FULL_POOL or (RUN_TRAINING and TRAINING_DATA_MODE == "curated_pool"):
        if CORPUS_KIND not in {"instruction", "repository"}:
            raise ValueError("Choose instruction or repository corpus kind.")
        for _name in ("BifocalGemma-E2B-E2B", "BifocalGemma-E4B-E4B"):
            fetch_tokenizer_assets(get_experiment(_name), Path(TOKENIZER_CACHE), token=hf_token)
        _config = json.loads(
            (runtime_root / "configs/repository_study.json").read_text(encoding="utf-8")
        )
        _foundation = FOUNDATION_POOL_DIR if CORPUS_KIND == "repository" else CURATED_POOL_DIR
        if not Path(_foundation).exists():
            prepare_full_corpus(
                "outputs/full_corpus_raw",
                _foundation,
                TOKENIZER_CACHE,
                _config,
                token=hf_token,
                encoder_limit=CONTEXT_LIMIT,
                decoder_limit=TRAIN_DECODER_LIMIT,
            )
        if not Path(CURATED_POOL_DIR).exists():
            prepare_repository_corpus(
                "outputs/full_corpus_raw",
                CURATED_POOL_DIR,
                FOUNDATION_POOL_DIR,
                TOKENIZER_CACHE,
                _config,
                source=REPOSITORY_SOURCE,
                token=hf_token,
                target_count=REPOSITORY_TARGET_COUNT,
                per_repo_limit=REPOSITORY_PER_REPO_LIMIT,
                context_limit=CONTEXT_LIMIT,
                decoder_limit=TRAIN_DECODER_LIMIT,
                disk_budget_gib=REPOSITORY_DISK_GIB,
                progress=lambda _row: print(json.dumps(_row), flush=True),
            )
        _rows, _manifest = load_curated_pool(CURATED_POOL_DIR)
        _kind = "repository" if "snapshot_policy" in _manifest["report"] else "instruction"
        if _kind != CORPUS_KIND:
            raise ValueError("Selected pool directory belongs to another corpus kind.")
        _limits = validate_curated_pool_limits(
            _manifest, encoder_limit=CONTEXT_LIMIT, decoder_limit=TRAIN_DECODER_LIMIT
        )
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
            mo.md("## Full corpus (all six configurations)"),
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
    TRAINING_DATA_MODE,
    CURATED_POOL_DIR,
    full_pool_status,
    prepare_curated_training,
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
    if RUN_TRAINING and TRAINING_DATA_MODE == "curated_pool":
        if full_pool_status["status"] != "READY":
            raise ValueError("Full corpus is not ready.")
        train_data, dev_data, training_manifest = prepare_curated_training(
            get_experiment(EXPERIMENT), CURATED_POOL_DIR, TOKENIZER_CACHE
        )
    elif RUN_TRAINING and TRAINING_DATA_MODE == "repository_tasks":
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
    TRAINING_DATA_MODE,
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
    fit_repository_sft,
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
            if TRAINING_DATA_MODE == "curated_pool":
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
            else:
                _budget = (
                    MAX_STEPS or ((len(train_data) + ACCUMULATION - 1) // ACCUMULATION) * EPOCHS
                )
                training_result = fit_repository_sft(
                    _model,
                    train_data,
                    dev_data,
                    Path(RUN_DIR),
                    training_manifest,
                    max_steps=_budget,
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
