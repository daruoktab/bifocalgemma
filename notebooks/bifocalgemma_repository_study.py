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
        "0NxMFF|-%}05Vem03iSX0CRF<FJftCZ)0I>XJu_|VJ~E1bYU-Jb#h^JX>V>WaCx;nYj@i=mfw3${{xoU-Juehk?h!YoYCZ@iQ4f_"
        "YG-VBI&&NkOOc?(nj%?}itTpt-}gQM5+otnnX{`tECLt6#l?N&QjX)iSw~47EK(`rb(-#kd0DJ0A)ztJu0;^7lB!Db?3KvmCs_)4"
        "C&P6uD^ca^GL$0BBk4Jg^YvF>f0e9?ysSk~)y?;1P%YDB(L5)4^Yxc1&zgr)Hec)fMrO%x&=8mT3KmFH8P<S8B^1Qle4W*@9E(WC"
        "!8)y@B&-#55!ASzhWr4Jh#B=Qk~j~8^jfY~ffogJ;N@|gCYkiBxYi@o#{E(PC8DvYh&Fx<ZL%!NB&#>hR>~r;k~%N%MYF3S_mfCw"
        "byD9W9_Ob&p1qy>XY=pQ#Z3JE^;ZHu4z5`Q;Z0mSN4k+Zk_Z}{Y+aP;^Em*00HE(KoUs8?%1;TN4+2gOzFE8o4-a?a7cXQOOkeKp"
        "zKmrQ?S8ZWV(%~x_rl4+B;JcV6Cmo9to<|zWmZX?XO+|%<g@zgg!yWvvGj}kdPz$TkzFL$$P&*@_>Bn1;r>A^_W-ZGNq7(i!HemO"
        "aCb4C?uLN-;!w(Ha=0%K{vCufc~>JQr4`Lp5|(+D$My3QfP0H@?|9yg$iCctA$RxUU^forZnPUM_74ujz4$Pm9!{qFi-YMwu)p`s"
        "bpH#H{rvS;ALeI29A8|_-}!G(-_8B^$8YB+=W~-oj4@6CL|w|AU2oDfEvjf|cPC7PH7sE^?)4j!Dr~-o;bdoO%(BijmW&qu0}@6e"
        "?vqqjE`A>!sR532e4XVbXpX#G1vSbIojdS*H3sIdgCrAa0-_qEl?bv(uq?z0D3Yp>DG1*+D-WD6p~<3l+xXGiz?+`{agI;EA6xt0"
        "{qW}OFY_~7=iAdCewd$J*t$;MTU+Npy+1yg+q#ZV-ffub-Q4K;%zx_g{-^~7daTP#)axRZZre<NUR8r?x$d?CRR!p!tX<?<W<blj"
        "IgKRu&S~KBDW44Jpddn(HtJ55#IQzvUr3SVG^L_-3#A6gyHFOjaGNDB@XNd`b5OU(86IDNx>&WG-^24AJ&ZPOwUi!c&nQS!w{$+<"
        "KKs|@KR@1ntgf6fan@+m<FpEJ3t+tLFq@iksE*wMdk!XC0(2_5cPws`s9w$v^~5|9m~Jo+_dgNmOVs7a*m+i@0T{iw46cdGMNl$H"
        "WtOk6moOPvze)o5LT2G|6_ht32eX+5g{PQ27OBjbzST(lQS9vw2?qw|Hx$I$9d&l!wzor&1j@>K)iT)KKX9Fo6UX}{PxLaEi8vC8"
        "c*a}58hL;lY|tHzyrsO0l51JPN|#5|{j0H<9vHlXESMm_lmQUjH8J1e*<jYG8ldP78mFoRs?<~7IkL@Nt25?zK?vkr%J7Dj0?oLi"
        "PD%Vv5QYI--n>9Y1Gc#r3os*~7#8>9{IBmppLn<qPWNp?<zLsbOu#JxWhc;4xfg0ePyCeB@O7ui^Awzzpai!eS{EV=GJ+0xs#TID"
        "tMzIJX^DMN0=ip@rCgUuRVU#qQOQ*R+AD<9xGfX(fl6or;^gbP0?j6>FfXLnY*ZtOjOhC?eeL_k!BsMi$KvT&)XP#<%RG%{6Yp@`"
        "uH7*1>?F@5x*|F7o%GSiQB5fBex9ZGv+qE~N_7rL-9@}+0pRlHdk<!~DvHFQu{TQ6S%Sj;-0LtV8T2I?oKH=P-f-YtVCzv2GFku("
        "?$luSxI+h#R3v{0bkx&%?1NRmq6OO1hoL_-ct#GQYfd;Cf_Vi<c~*O?n<yz=eyC;_<r)q^-T`y?`3*e|m2PuBJ>0rGR4963o@L<b"
        "y0j=9Q=WbTA@zdF2Oe}sAZpI*70A`B%t3iVn{8G6kmVqs9=OwW=#*fDgHm7N7jRV3T361<f|w>V<L4=l-*wLC@8@qX#M{#!PcGc2"
        "Bk|qY=?}sItZ1o`7t1;Xw-8XioLqrsn{JpJKV^`B=>087*OIJ<>l}klT7kU)nQL`tf~v!FEL&ua*lpiV(cUCa#KjVr2mTF2lvF?w"
        "MKD-#tb&~Y@VphuhZ8{5aE<0Sf@4SC@~q2if>mLe6tz1<-FbI*`axX0`S$%>9DgU~e?LCII4AQ{y%N?weOw#5>3m;_w`cP=7jxB1"
        ")44kK5Et{mUx*K9$KWRYL;Pj_kFmHegH<KIeSiAxs5g|Ra~m^k8QYn*KU^d0;tJyU<YN9kIA)z8V`DY~Yr01x@ze3epHF|h5ND@9"
        "9lv|!bm=!Tm1+=0E^AReT)7km<7q8Wmun0X4F%KeZ)Wv7T1ckU%;HQKK*Tqs)2bYm8*Qrsqq<4GUNtgiQN0qBIj`H*e2T$0E4_ix"
        "9z!H?b|FsB#PRnhr)P6KkW)Tl@wYece}pi@eLa3X0<F*<c}vhyPihuvPVz<M^f0`0oj;-uaQ^Cur{vo3#=)|62r+szo@et9@87&t"
        "h!PtQ5N(ee@p)iwFxvMvDYR2Yb8m0W;dqxBx1|(mUE+E4ubCZ2;{p5XuXx0^!=t0|K;J0aXgw9nIL$kk8t4knsmDV2v0%B$IB7$D"
        "iSSn*ikpE!D~AyKg-a!g{}1x_D8?`ITja|gTUm}>S+PbRkriWk`03AcV1ova+3T)Q^r&tcyl#*7iU23Hgy;tVb|lkXXU4(<daS_<"
        "*V15uE?pRU3spF7_}#gIqo!OEM(Qa9c9GwB1*ME7_%tDAC#VtHs$`X<0eHFh&^YxbpTATc#)J~V+HvYqqS1jkbMM&x&lGA3zM7h5"
        "5cm=jm{&vuRD#j{QYJ*%%HtTFg(ha2j7clD8wAyOIGNZSuW-4}ZcMkkx>U0r>1m(g*sI}Cic8V0mnCk%8LI%$d9BqlP2N|PbRPL_"
        "^(fK7PhWh-U2m|_o-9hr0{|D)KHIYn-kV^Yk@)`X^v4h4+kZ6c+B9-QLQCHUC)xBH;K=FJfE*hfSLP2x!9JUD+g6F@QQo@16Sa`F"
        "(L|<ZHsry?6T3R9nUn(?BDa6I{W!kat4}L1S}Qm)pK}{-pC=#1^Z3zT!56U*>k(obw(`s~8co)|4bqka>_r?yfV^IU6OVX-0%eb7"
        "oeg{z%4(1wA)Y@Mha<aFVL90D5>^x3qbDkriZ3%@Oyewx|K@r6X1&qD0Nfe-oRMZ#1{=W#o<Zz%AcqmDG3lRXz4U-(f7*a-I7)3+"
        "D21c7t^W(Ba5l+k$KW&vyaBdjh}s7WxM89?CZ_dM!ZfcW`6(R_<xBi(zSig){HB){uy;0@44s>7E+~9ZIW7d6lni1$ukkBNh>2^|"
        "P81q5-vB6yy~Di@aBS1m0_vbyS>mKhvZ@Y%=&qg=L}G+NZLhnXf<esm<REHqH$nG=*|pD^?5ekXDLwWQOXrNUX9D<DTqmDoCiprD"
        "H;WwoaRGv_ag5X8x|+f0;|K7_Gkq(i7LJj@kXjO)>7b;gfByMr_w^CO{j0bieXO2-%v|s3>(R#y`rz?DZEK&md*NAuPw*^}cns{3"
        "Hg$0UuvY&u6T6S3TCiFq*Xw*;`AJr+YoF%VTUf|AC;ymQz3Vbx7w(kuKp!(@QuB2U<9MHVfvc$ud+G1?^>Dk6$y)Xus|D1Sjcq*<"
        "Z_s>$DNz^4kpv{FBF`$KP>`nit&GNEmE3X4Bn>jUJa|s~X_Dt&M;3{OY@jq^eFySU^klrd1jSDf*QBxA>AA0`;58UeW2Zfe3|bWy"
        "OXdXdPx-V!no3X<S2j@O$@RjpkEQd^hr~hzl_&ru?2mJ#E#A)?^M3T`K-)<j^T7Z#T3hiNH#z)Aq5D&VE2{63>i{U5pXJIZ)^I3v"
        "d$nx&f^b9`1Ki9-)Z%mvR%ivPGIXlYU>Q_2cbQx#S&$0jwjC%0^g`w3i4#iPp?d`GTtHAF3syQUKeT5K7(PEvQo5V1@^p>(ctyv<"
        "1O~f(5C-Q7b1=T(RskFfZ+{X4fH=(TH|T+e(RyI&9v=F8F*7e`<{KRQoilY2@vx)Qx-MWz8LZGCIXdKMuQZ!WHA;f|MF8g>`FuQ;"
        ";tusR)(T7|a(zy#a_dvwo91|};}bHf4Pw)v(l;JMFuzLIz_6~~C&`AaI`DYK!|NG0C0~DK+*+T>AVT#-zWytYd7rnDG1Y*mOCsdx"
        "3a>mv@qxwh*EK+a_jHZUd3$*cE{`nuQseSU^V%=qg>{;6D)UaHlHfKWA<ahG0OG)*X!jn?kV5m`=;glk;%Iu%yEf_(rz7-=(1=N="
        "aK*hF>vQ2G*By#==kyfj*vT9)=CBgO73PSZ!^JYKE~AaAm<LrokMv=rkF`lP(D#TgA-Lm_(R>@IlkbE`$ZWl$%XJqq?}tJv^m72%"
        "p}23_?eAXo>yN=xGn>E1D9n`4XAovKGYeQadjTV?!8=UdEn3<TceTij)iN`)^2k^n{1<Dz597f=j|#E1^OxUVI>>I$l_#^Px=re("
        ">!AMJbp5o=!6U}CT?r|D=cBq;*|$aqUTM?fp;J=!eRPFvX9mTb8;WxtjX3G)W@3|Vk!#Hrt+!ktT&O=btkxA%y~kEu?ZX-wM#El!"
        "HUoe0a{LM_GKgqw9j-Qv20Gn!4Ym2PgyCJrGT55k1um1@lyeypR0kTZ1K_s~fQ>IGz=J&^{0L%gt>=IW_NxI+TCC{9{FLiFN0(&>"
        "T>GBlV|zpdB~OEB3d>le<I($Tji#|fRYYRj9xhJa7vA;K`LQqHT&d$EM+1&?^HnbvjIhfc*^(Taam~Q<_ZRoImF{YC!fiI|;Id^A"
        "yCyY1L4`L@Wi_}c=znxElToEKHqal7GYP&-rOTN{=weI!_5cDyI)J|kA{Y7K^FtV}6;(y+0@qN%D9vo2Se-5ide380LGzY<VuZ${"
        "zvSSy@6TXC(q4K2R9Nc!C8vI_Fdpi*x5*ugD$pOxX3z_wo5ewamwg(y=Kyy&e>Cp<#~n$qB@?ZBgOH+}tM=0#EymzIf^tW1%)D~}"
        "drRbMC%nY}+jK2h;6;*=>(JSo3PItIxhISBfu&9}!<%fFiRay5uL1n=Mj7XBH`?qD+dG_KS<AqS?(x||Xm$eo;wrajEVf|Wrz73J"
        "Z<@*KK5b6S<7Y$S6F<rog+R&hUoW}w>JsaWuQnt#jnV#%q{JpW%@2bc1<BS`oBpk9537=`ll8&2&T0BNFrn$#Y#UbQ997rAUD>9L"
        "F+pyNPVE>4;4}hfroIk$%}CQagk`K%8^buer)IXyWr#hprjLb%Fdu&3HcZ8rKE{`nZlLJ1^?gg_M#k$b<N`^bY!7ADHWBdiqZA(P"
        "P3M7qWZ*AS$TVCDh4wxAYzys3fX69d9aDQjQ9#^)3t06ehh`eu8lNdwpjBSmY&~#UMTsxe9R##xSsmq*vf+`+sRsRn_fj@=coiO~"
        "clUd%P;?Bvf#)%QgX<7S<Co9Fdmfz`At$>Vz5bWjpG+p`l1(g|*8{Ei@I=r;6`TAtGFPVrht?oi;Az11vm4#(5|x?Lr#SB9=oA|@"
        "%hNL5gccK%NWx#cske@yDsuROdY4NhML+k$8JQwoSK`(v9#*Bv;!l)t5$^P$TQ#gpm@U=AqgQN?2W8<{f?U~4plO{$qwd(ayTh<4"
        "lqJ1O(Ok3{IXdD%UzI;nu!5O^%2j{qstIgoG@E{duTg#S<1!$10!vOisj?y`0}L@Mu@bAk9ULo~=_#3>XH7*sxP}P|JSYz1^$t!3"
        "OvE2M3-$^{H~ZG+ARTUSF)rSvVrfTPq^`C(8g=TF@HGTCG2@`yWdii(SuT?+3{aDvOthUd{k>`G-|w~{iP5DvJJG11&tjze=x(0W"
        "WiJh-_F&)x&H`lvlmR+woWOzIGEt*Z&u|#pKc%N2SnWiuI7{DPnv*;cZ<z^@)A9AcWq|MM1o#>i1;97ebCv27t94cD+W0z$fM_X8"
        "&xQ~4Rl5z9+&0%mBw+yJh~tJ83aS%{YSU(X9O~rp4CyOEU&6?~W4hs%zX8XKc<wehzFWqQoH+X8lWnJKa2;Ab_3lUui|>zYziEUc"
        "MQ33Z+(<u4Dy#{reNqluga8v+nuV)2q`Xf!XXTm>571+l2CGFBaLtU(Q$Rvfx{+os``!g{pf0)rVeMW*aWD;tt%5suiga89IZW{u"
        "Pdlna$92$V<lM=#G@fZum7G?aH%J&stY8Y%%@@w3s19F|+zKgFg-t6LV9*`_qJto1ObG+)5}v%P)frdqA)}gPtkvy2tg`i#<*ERW"
        "vDNmS1E!=rHncf;*VX<uuLCg_1g!FtL$+XKGNz)|hRP_;?1F7nOOnHg-#RwkytBaN#wIO38Be`v%W3U2CEHj%_*&TkYZ-fPnYHf("
        "J4V-0UxNCuOcw@IwCO=0osOEr@Ao2B=%C1s1EL2utg13a=t&u^#)!|YENGifFOvjt;<xAyR{T<fIuKQ6TijDW-4zGg-k=KukKRhI"
        "FqOCp3ap0W;pj^`l=x!hePkY0KM{ur(4{y`mK33v7g+Q|4yrr2eo-DbfWASLj@gR$%W7zp2h_T8C!<HKhL^CzwLoUTECn^}ibsv{"
        "t?8BWdi(|EJ>>9Zm)_fLJk%Qwv}I236PZSw*i+*$@kawdd3$vZ5gR!r+e^L?1scSuW1MLFVg%fsUh@ai##e9inY_VGK*-)}f+K@H"
        "=shSJM09qYa=~n|1WlL9o<iV4RV*OKdo?x!M+%f)L3O8--4~#p@edZ%IRn}=Qabs<#i<>op#Y2*6x*G0xfw(%#148Gd$#<<Z#e2e"
        "JCf$p(=Lpkf?HkBb|)_myWqo)MNmol!2s@gaWK(6wu|T>!fQE-(A83wyXX(R3H8q6I!k_COH#P1_$Mx<{0%ETu-=HT+@EVSz(C`x"
        "sjZ7W{9yvnk8yIX)x-SnALeJr_%^hE{$c+1eA6gWeE4p*aRXueMuL7h*u0=%@@tjTMsXaCRa69hT`$Rfy&9|k{tY)Nqq34gjd<)l"
        "dGbUY8`Ut`?`WL<V3bk(;p+28Xh*>BfaFJlA&~2-iyxWp&|uUoo2sUkD|FtUJ_TrshzTP~6wx;$)iTteD#^W>fVTQzSkaQ!>7V80"
        "#Fn1x%7OXm%7kT%V)E(Ubn?ya<_Wfe^<N&+eEaWEV7TEIz(f2Rtlo<qbt|S@5v>5Ts{1<x`s1i2lna0d>giRJXy8xK5K8|kB+PLd"
        "ucDN3a!13JFG=0%GKrdZ#MCx(<3-p8*#<Nktc#9r-hxsCvK_5nrZI^vYVd(-@*oAPl41IUYov!0=oi?P$c=Fg@IYml01UsN&acMW"
        "ZVd2jFZD6{^WAURYwa}K?o;{$U+};N+lNLFG=?($=5aem4_~D-btFgwhp9e`#0*ycQ&%*GBQQwdK^<<zksa~bHgdpRIJ#>ydtzrz"
        "zLLSvdmaMdXWO-m8Dbt;a@t#OBXnL>RkUN0E>82?trrvOl&C)aH@WG4;o%hghU5e7fYH$_H4JMJo~=3aN2kGir||p3R5gg#rp-Th"
        "o%Bvp+D+)UBetZYja0Pq^>A$(2Fo@1epX@nv03xiHD*n_zd^vUJkqe|(Z=Lq<ImM_H(1jiGMoWv(<FD7H0G+UrcpzCr`n*Q;jLyR"
        "4HhyzQj1(t>I65%&!lHD+0i6AEb0l-pjpNZm<sia{Rss26K|iw{Nm+4J-yVAFZuB$J%WCM$7x@2e(t#D_8h%&L}m)aOv7(FvaOl?"
        "PM*J^rp^M8Rsq<iqgw$BomKY-ps1rY;vKcvkJx*LyQ2vEuolbORehG=o(5LyLYAMBN=C|bp#-g0m@Vr`{iU||M&_(fL*ovBf26?2"
        "^h}%g=X&-rC4S51J(DueYyE2&3^8guw*T-2{qTrRHTA@;Nj>rJQ0SZ!rBF=WuYQb!uy`(99`Z~$9{xsLYStF1Tt>ZAVJc{5K0=6A"
        "X3n~fcMcu=^@dxd`wZ(O>`|=7PN%k*JQnTCFYoq~Tq?`8p-|$9IOE^GGVYbI%9IoUSBqg$JZjG<NHMSjZ%8ik{01DE08X>?>=WZ8"
        "R`sDj9F=UbRlOqhZn@Z&alP7gv-0Gj<y76gQ3U$XLXRF{v8nh`i3ob1!bKMHqV=k%xaJlFYz=xNsKO*s%&RJD%=%zb*u-a8z`*wg"
        "z3d7@uYxRzIlW>kL;giH{z1Gpu=dsOhmcJApQfGads-7l#C<KsqN>Ul%DUS{c`ES5hf0n)R9tS!x--`kk8j}ctD(qj6zHLT``o3`"
        "rDNO;Y$`6$z-l^IDFsKjlTL6(e7Zim|Avof9JYoZ)2|$4^=#LA_Cxi-vvreEda|#-`d?5>0|XQR000O8001EXe)w6WHV6O!zZ?Jn"
        "Bme*ab8=%ZVrgb?V_|G(Wo>O?FJxhKVJ~E9VP|e{b98BAb1ras?O986+cprsXD0svr4Jz!iFVvhr&a4_I%(SJr89kAWDWz7kcFBA"
        "SOB!;xc=|m1;D2$S?=Rj&LkGe-NimX?1DiM{KB?oE=6%i6r(v)B<J)>NNsqgzaTWHm0>xllr%Dv#f%7c14MGY<{762Suv4qN~*4j"
        "&ZJ^G4uatMv**uvSxIF;Cf+=Mma{d;U=>v?g+KBSni{IFb(mB5%gl@zdCdg>z|@SqoRN}l(@Y8j%BA*=zWC(q^G~D2^JfIEs$La5"
        "n}JAvLk%MrWSPw1pL@HOir^xI3*zN3gX+#~qzDo(!AhaZXv^C@1H*tEpWZ)oPf=n;&1OcdQo^8U4F<A2#FdtTe;y<;H>}iQ6!jLu"
        "Yva6oLDs>gI5{~X-|~Vj$UTaExD>&V+N-g|w+S%mx(o{@!YXpQ;QChmh`b{B!ikTDQY=LbK4c;fJxOaq)FWQkMFFb>LHxVqu!@yD"
        "G@px{Z87{TFBmS*L|rlk-uE7GZ*-81E!VPE8OCVjsiot9(14LalEjcoGG}2>oAvy&U}VdRffgvmiCZp6!>$deo38E*Lm@dkAVje>"
        "Q?N`=&Xg3=2?x&|C))CZ*d}r#<9wK$6(d%@wgFf=5eqNMF|-;-3Hj)vGhpOnYaHd8k>6-hvv*2K6$U?n!+-p`34pGsASKr&HQ6Q}"
        ">(7NYs?H3T0;~vk#e+1XI8;BOBF9)VYKYk|!nQOsV1aPVIn$ZqD*%lZE9C7ANKLH-7gxl>42SAMKgK&rkJqc#^G;5MGJ|AD80{&o"
        "0PE@}_D9XM0qOgM!E>m=o#j2=ke-)!;xyqGj8Y3;O%8jNUNu5zfTVe<D~6!iAlfggW>K^&@v8~=8#&-9;m>4Q6%0_)*Dl*;@I%Aj"
        "3P6;b*6>xGUoo?r8EYb>>6{WnqVOtwk`An_%$-+YXMuvPO$s^0Z!Ajit4;3~owQ4iIEp2+u(guJ8cZ<-7fgrO3{q3%-o2yxN7)VU"
        "beWOG(b7)HJMY37LFzdz_?3Xf2Zy%B^aa6<4r)f<LPBs2g@*74Z)kzlAR70_L*a<IIR_mPt$<2Vam7MN47(4Y52Be<9c_ws_8T9$"
        "zHhN^TajOqR?Y+`UK!By+-<K&x17Hw^HxC8CHFM@bhUIrJXkf&(od?J=KAvD(E@6&eYRWsTRdphnI>k1p2FkK)P8z@OM!Zq_<)R~"
        "_`%zQ!kLJ*OOzvm-5CDLiCnfJ|3A<N13f~nr$_oY0UnRXm2EM)Ct%MSD`@ktCxz|_;=jMY;q}#2=m1_mKHsI^7rQ;CoRFW5tcdSU"
        "1cRAF!{#PykyB`3V5)I5%<pws5oxEHk{g_IEab<N&S`526hA*ZJ3C$_o+I3^xo`g69I{y4+iv&;{{JF*fP3Gnyb!cx@MP4SxQQpG"
        "74Ak8k-DGOC&0E2^-g&nEPA@k_Rzat5l91eLyM}Y^{JcJg2Nmz)!lH7?G><|pPhd)Kl^lk{_$QeC5v{Iy+pWkrX}PXQZ@}HtYrJp"
        "1AM(+7LD%i_A&T01mc~<KFpC;62fwa`uR~g{f?C-O<gaQ7HrEid8KHz;aQ6LmEOW6C2w_l^LnrT&2*hMV>igRU@*b(Fnlzg0NsH6"
        "-{;NW4OQEC0bl$3XfxetF+L91J#xsD)Ea@ypE@WAz`uXwAa8A>2C{qTV3)?WR^G;t5n*5-tYP?XRtsk2{vmQN{e94Hk$c;dm1U<F"
        "9Bx1)%$Ew<d()qgY}Yq|*^fi@<m53zy4@E4Eo$`^R(J=21`kxVv9P*^=?}105)H-*Kw|}IK`}1|6Tko;p;AL@R>=&-vzs7Uy4-?1"
        "OnO|0@NgohpOVi#45TIfAjMo~9MkeN%l75+6bfe0;4_t0T)<3_Vk#<`F|FYS!#^qJF<2bdnZ*-7-r=HC3_~mo-ZsOa+Z(sW&f<9F"
        "U|5M!qH6<@z_+@tV7jq~e{9XSF-+FB+UyjM1ehmW)Hb9ua?4?3tBpSyz;PdGJny7x9qR4}+X<irk!W*`U@I)hc-ZLR2A-!~x=k*A"
        "EXMYLr=Na@aP^<&-;-lJ08y--D}~(-cmU6y9yRDO6ctjHFxq|qe_J5Icr>u+k;x9Q{#?K#qCaL+EXHZu9y*08c>n%=_-5gnpg*g-"
        "Kdr1cm-^+U2;-M;qDui3fPfCEfca-H4-d&k2VZ_|`#46EtMK*EBm_!#q={~fmDkY1H9kSS?yKIP{IPRwJYdhZ)85H=i1?||S1jkY"
        "xOC$Y55M<nIS6{<_6U1$jJyvKp64N@z3un7yAb<Ljpq!w?Q}ZWa|PW(>Ep}xbnO<OFyIL)^tf9^*cR-;Zu){e>I28Ix`UTf<dDtd"
        "ZE@P1)&&}<t+)tgiAIxycKs@@g}fEhR*~<&=?sH0qAuJAgHGV6)<<<dHfU6e2xS`2!gndrLTkvF#h$jNwww;J4Y}NWIILnzwj2vc"
        "B64xDLkm<L#w1Zsr_o1}F1GG@GAPHMvv7k~JnJ{k&NADjIrDnUba35%a7t}aPM~%`VVEmXB)sejtWRO(+xPfk1+43PwU?zn<OALw"
        "X^Njejut{rd)nA!#b5S3fczlEz=V=(4O?j1OLD)C9rKrM<~*XTe`cF&`>X%5;4t^nMtJapB+UndJ7MT{uHk(FBeTKn9{4x1$Y*5O"
        "?zN4>DKy%esZ-dhY#-62kd;sH15(%Lx!CKN{0gSm6@V|Jr2|>zzHi}Hq#;6c`_LX?gCg4Ej5h%??=MhG0|XQR000O8001EXHbN|u"
        "G!*~<(?b9NA^-pYb8=%ZVrgb?V_|G(Wo>O?FJxhKVJ~KNY;0d}Z*OcaaCyZ%YmeNxk>3sYKj@STqTclTH4mdd>>6X7v5#!*-Q<8q"
        "Ltsl(w>)Y|D@x;;oy>nvRgsh^i9H@~aJU9+OMDcI#d@=3*LB}jN#Z1Nz*E6PCr(9KR6!YMsgq|}GIaJl4UQ*P+&aHitYB#wr`%y_"
        "=oCE9L|kUY!$@RR5kPAd#^u;`-Pf;Pzlu+JR+P@+_c*U#K8Y-?Urww%)-NS1qB!C8dsQS!ydM|*w~CAM^{c4JPEL>|2`AiyqsG0>"
        "s<h;V8ju4>n5H{_rWI7*u{s>$^boOt`^TytbByrK53|o{lCiKJzK^3UV99}>PHY^ql8u9^U_=)+(|i5Of#1i!{rchUjsM~1-N&J9"
        "d7Fj&Z+E;%SpGf@`MqkU`P|Q0T=3ABBonHS-y=w$C@x!&8h#ibVE3w};4Ji6QDmQe_@ifvJc&y`$kG%j18@r-WJRcZ%j}k?eqLk<"
        "4%!B1cb7#`0w?Q>sMH%2CEEiM5m1d6c@d}O+1U**=*fhw+@wU8_{EPy4jfcIAf3Ov`RVQd{P^vM@BMf0f4KSRY@A1lwR@~iEai7B"
        "iQ;?rT2q?3_VFPA`cCR0ISX#xp)ul$EoT?&d33R0m%)C%*zZ@Dv)TS)5v~~9&(=#G1<~T-vNb7xD34h>DS_OhDgH)TL=UrZ&I(|{"
        "QV$=ff8BtkF;F}%I~b-_UL=#ZEPY?zV6VB%BA8x8i_85ix>{VWqQ!i+i~`QCmf>=}UIrIeVKfaFYu1{C6#gpA3NC&_q};z(MFI4l"
        "e7N~I`nx*>oTvP9HQmpGt7$lcm;FV!*w3RyaK+e~2h*!y9!!@J!?C!m;N=*F)<xJ)=8^sS;Ro}?@Nj=)u&8@1%UoPfCWp8L!5RnI"
        "X+p=8RAM4W0gomeKu^@aVxpyQGM~<uqv?7yU#PL8riBn0AAWc111K^fkQorZxpcukkLB`};04G=2}B@F*opm~r6Uo<pw;*!2yp6T"
        "IStmA;e3A)usOTrj9;z7tLcmdms7rGtL2nkF4<}kPUs|B%lgFrpuNWTgxC`@_Ul(6kDQ2?!O^ePr;j2iydulWq4Uj9ist4WO9UTW"
        "D+z}_ped>?8aHT~$EVvcE<8AK*oxTvQdE2hS|b*vpWV`{8jLDX;K4j8J(3m9Eq@p~9ugF#@Gcgh#fGF|Qg-44Q2I_s7RSKZC*f(t"
        "YQS3nwbLjzefaAr(oU~bV1PisS>s{pB9&1-<b0#K(Y20}TiVon>yHUoa72AtY>3DcV3x=ROP&h--RT`PuY(S0!C19vh{AD(gg`un"
        "0n8c~q`p0O0s=Q1Y)7+QGdhl(6b9m?F`aZ97-7p%*)q_&8#(SXaJ?_82=t~$4f5IftK*`!C}R+mK^sQf(vIP^;AIi>JMO{ruvuep"
        "W^Gjd30PHHaBzlUax%s8w92YTWp5)a`zlUCLs()ooCFaphbXJ?0c~6qA4qltSQ>UfTUs7BKV@knL9h!nudpI8i`U`mf%C6UGai&e"
        "8OV)&wB*3~uT2fhpn1V!!JU7wq~bSVhzrmC7M!OjzT<|Qgk%OUqZIJO!OKa3b<r^HIWP=3N#Q9L$faYq%SY1OBP%@`>c3H~!E`kR"
        "4F$T1U4+r)Rd^K!vuGA9_MpV4>&pcX!}*l0*7GZ9T15e4e7OjDFq^Hy_2sN(Rg{r%my3OH6;0R6C0|Eux>#PYV09S*aK2we7yFC("
        "Vl`i_0(P<IOMrh7tmols#e;pcSkA+_2Kr=Sc%Z4_$9N4#5P@ZDho$=@ZkR$-);}_F^n)V<$0WRVfx7{wK1O65IR35^=Cydnd{Glw"
        "5~D3b&N0|+OXT-er>n$spJc&p&1LQJs?R5v$lWRE0s5&Hy^M$Lwf&u;=~|f7L^lc`B%lhsu6t$D){sH)E6FW{Y2f(adElCxO~=J@"
        "I9MTPHeI!j4%D)Tu<5Wgq%R&<mFl5}IMen!C)JQeD~Wo+W&n~Y9^%%KQar6r9KC2ctb+^MWF(Nq^lV}L)&)QKl$Y)f(4hA{gai*j"
        "#iT;N-rWry>o7QtERDh8HHPl`Yn&r)e<V>`q9m0<y)=s;nCU_9XkMhrry*^(O%x(PmnwuBO2C+fp%=N2faKHlV+?;jJ-HH$S^%vF"
        "1#IIihog`zrS%V>=oVmqVLg}vF~P>I%4l@y4jd*Nki{_J#?ol7;Vj5!BY!}S&@K%M2Xy18t#ZD#a1l>H-?R~wSVj*W7C;#0Zk6QG"
        "K3l;Cy~&N>Hk2?AI&4Nhh@-qkTuRPT4erjO-@3Gwq>{u%aFV_>zq=UcK#gV<`40M`$I}v>O&6Q3oB_mxQv!||)3I<*AIdo-E&i8@"
        "wuE~9NQdxb^N<t<_4*JcLgc!CD1nQeYezTpG*pedF<2lIweyZubNqh=g;m>S*#t^xHl03eji1pyii=z7xZ!P}FEy!{I;jpqDR)ye"
        "RFa}49qk}wN5a60;0Qi}f$E8$_PhX)s%w1UrEK$b=t<Ex#3*^fQN{ShVdn2o^A&|XEw4?Kp#4t<Fn(VaU=|TKPeL_DGXXJBsVBvP"
        "oJ#bWI$F4qIL)Jo5pt-U+Ex6Zkqz+{9**uDCDM`nUC?YNUNV$+;9R%K7PuWv_b2DqTw$u!45AD%1usTs!s(b5B@pWrmt*Hg27#6n"
        "C9DL-<Y5;gsP8a_rMc1}N+}HKOtT6_{SKR8DV93yw2u!}Rtdl!yo_;wM}p+Y_hq8P_dVE;I{;66?-lN7I~`rIQM7x!7%rzz|Bcoh"
        "{@^^I(`PvE!QiW8v&COpF0b*Zt)Y<3ejN&!6qZGWib6Ib{{i0d;Yhio5;%x3{eo<2%X6&K+nzmEvOo@>hPjgqq00jBx8R$6!8+Nu"
        "OABgyh0n-CGDr!1YN#D8{d-Q0%-8FcgA5L0Oeabn&hldRQsvT40YG4SEBvhR8wQn1r>ku`x}vRg3f2)Qim_`O49@y)FebMTDeLvD"
        "a1+Kr44i^iP9Y=%pcqyJWy_DGl_*JqhePNCVe1AGQ+hXmqSojF#$z35LgMiMSVzMyji5+*eE|7oRvvkfnjmm}#?XI~xsM(L%iW)V"
        "tWgZuQNG5`wQV9(ZdFp5662jN3052K0Y<OIm&5=mas7xJ`b!-CWv6{L1ZVgFJ!6JUqWx$%j>gw2$|hHbKRR?tbpb)3d@!TKHisLG"
        "P4WU$1i<8>5+8GBEB?VbF*nAm-qwsdtP@8zU#(&A<Lz}TaJ(DHcxmX|;i-{OXm{+%1eJ1D_$Ranet86X9J3?@c#^bri#Y=}nRAST"
        "`8}X8q0G++!8n<W9-Rz*G)h#4MwE}VQC?}pPI2;JT#vi~X>oGYP%KW-t^n?WpRh9^{|P3f@-89ZxOoUJD6G(-bSqOr$PYE&OIxDS"
        "=|$OOQ0VfdbyD(%fq1Z&MCn2Un{fc@h*mbmRkuTQ56~no91#?pr$CtG;f*6wmIGBW-2uFxlPy*TC(Z?&TLN}ZUJA)gi7GiC35(Bo"
        ";cyQ@5SSx;DLB#GzniyW;X0mH7eJmN<LWZ&Os@_-EL{S3)pGkZepwVjt1S-g#^FS53TU^}zT&7wm3>B2Tw5poD~ku4W{6N{G%6}#"
        ">7il=jx<uS(_o<83#qA7=TUdjAos}vm&nmm&=BnU8mc`dtD+H$=0&ahT>8WU5H_b{JDpZR)elW|Vv*9L?#QJ~H8JdqH`ACHLPW{5"
        "ZDY#Du_UqhVKl(7e(EG{l{T06p+JP&-B=*sdzJ<8N4zdMeGe4zr3*_(gn~m4MF!G#@?j}ZaJ=y%%%9DE*a<)~n;t-@coGU5ED;|-"
        "8C3<R)K~Y)&(rq?l&vsuuCpbC_^#{KDv~z9@#sLlUW|BH<q7IV?qg(u+-}1iAd<<f*Rc%J`e54@BH4;02mdjJ5CeqBCX<97Dvd4v"
        "eQt!7Y)L?##D^FkWPZojO43)W`Ughwavv7I4|rZWUPFqZ^F4o{Ckie(K3S;Pcmbbeq7JidxMP}N-5H#Z*89Z4o>X%%Z#HoL?9B0&"
        "5%XD7%W`_6_vrR)4wzHQXS=n7fzFX9-v&=DWlYvUCjcMY>)7XaLKIhNUo^;?y!90Q-ka3%PH3=9D0N4YA5wcJrdQHd^*yveek+6x"
        "3bpjLE8wHSc<^B(ank1sk|iIGLI+R~-0%Rs^9Fd^IZy=bUZg}|wyByx=`uwl=?bVEcKc#d7a~v=sZeh+aiiZ_9)xUy@Hww^fx<U!"
        "7_ckQ6Px*TdD#Zn??Amm`48E%Tu*gR%P&!FAn!4{OW>HQrF5@Wi%1Bv#+Yq3m>1PAUBaO<GDt#)*kSSqRG1FcJ6P{BTg4di?h6H~"
        "j@kzL?hO;dV;ZXR?AF|UM{f=AJn`X-_t&;-<lft_(2T&zdf;acRyG?qCbTbk+=H>ahQ?b|T@)i%16jhxdL=Oyza`+O*A)tSSGr2Y"
        "a4I+U0vTfiTaUC#eXs-5bu`LwcE%oJqX379U`8h!O;vlpjCD4W;>{3>F$nJ0QwHyJx<nzCmJZV9Ip4_`UQ303Xy{7<sH_jU^%UxE"
        "+Lb^CQQP%GG`pn&q&Rh3+9KY1>(5i8T5l%F+rr;|eZi)QetnhDZ)y`s#)LtV39h|0QqmB~^Shy}t3r>+USH*D8b23gD;5SD&@U1*"
        "2Ka3ggTe2%b!{pcJ@0F+-LCN{jU`$kaU=FkfC{U06-X-}*Fp*7>SuERF%#~qsVr;ugapHYAOCGl(k2j`dW76H_JBrkM4+>-DP=)W"
        "ovK8}$k)!Y+rKBXDdOL`ZFZGo@>fWkNQ=NIEIu3uJWf1JmJn(K5YT1cj8{1??!Y+>r49G%y^HGP6<bLQAD|pcz<!4wU;z%ME03nY"
        "qmkgBfeak?N8rLV6Kd<&Pr&gB@a<976@rKHfWC-eC$43l071M1PWt^DXP<#xMB&rIE%fPzZ=CnPNCg5UE`W88>@LQ+0=z}xKukp$"
        "GnHBoSmH3sA2~Pk?<P0P?^<;Sth$durGc|pWC>+nega1auj>m;^^1qJJaSBwn5C$wH-UAA6rvf$hT4O|HZU8lg`z4AlmY<G2WEWm"
        "HKwpC6<)RTEhm=75f)%<%z8}89N-g^T2Ky;lJ_P(s&&$JUte{+hYs6I)~Drw5pY3d0dU)uMX@CLr6j?yhIMVD0ADqx!za&6hTPe<"
        "Bd)-ImeIQo)TidP){Eq(YSK1->ItCNcd4po6c`-rr{LO4pci>-COgL87KaLB;-9GpHYNgyw(*oVz8_|R?+?tW7#f0?Cg*tWXhdaT"
        "5b{w?85(ZX_7MXYTAFcJQe8}FlOQE|C45ZQ44#K8r@a_(Lt8j+|8Djr@U`sOHlXX2+Esv7IOJZ8q<irkvT}s1*VK_6PF1y_&n#ld"
        "-)KY?kgyITIo!DDP;%IOg*!O%B;U9<_Za1arOm57JUFV%&(URsf^w;(%$(00@2kYvZQ1cNTcvcUab;*nL)ZQkbl<B$(LV@Gc#VKl"
        "ZU~yjw5exrIDUn9wRzB*4y#H~!f<*{9)lMVp51C)5C+T9Qj%wE#eZ_qcc7eas8g-{fkzc)dvIAyb$u?t&vcmbtV62kA2+vBc*yNL"
        "CtB8YLSJy>M01EQUsR;pL};~w1dqN|CP{fG!YZszx$GZ0QBsMc$>jL+n$^+2)*uz5j)?#g{WiQI`F|-yp#w`rORa#=tN^bV+Q8uO"
        "X6a$4u@x5At+f>U-ATRBtDDT<5P)v$%OV1;anEnYM5l$i9pe_Z-<U%;rd93jf%xH%rU<%;0OOA0hj>pluyCC4cb)<FkZS{eghT$0"
        "s;b~l{RsyX)cf{__sz{j|I-E~A<8=4{O`{<AKw3X^V2W>$DeQBemplzR)kB+wLTa@5vyGgzoxBbHp#@QwwhiB=8@<g<ri2FNRR0>"
        "H)`dpHOgD1ITQ|Z;v44<{3-MXz(Q0o4fHqAJ2++#Oo6&d`DX_d0R&8N?2W#8v5HX0*11V=3-Q#iHKo3!Kwhduz80@e;tWlOy~w}T"
        "N6lDwXx5ew+Dz>|V!H4^-f+L%Rmf8eU7a~KgFsuUe*v^<M}(HzwR(0p3<IgfZ)ze8;<oDJ6ZBWkeSOk(0$q7QAYtg6q#Bf}b|N8>"
        "l})|*DX44G0yrz8X`{#Zdb$dq^y+J!P^0QVStO7ixVIjd(4Hu-XC$<`DL>f$*62rI`fCJ)Ndr54)q^o+dCpT?@y?gOh#<k^(bkQ_"
        "9vRGYI`esioL+zY5MjnhzMX~kvM(y|*L*QibF=-E3H+V^YyyA%+@|?z69S#nmwZ3e4fVwW)jwU=w9$g00@6k#bllMu6OnTaFnc<^"
        "lPgw&<l*qK&LYudRnyoj<e=Nk$Sku{-%3Vf8Ii}KzMz%JTCP@P$JU&)8#Y^McVM}P-L9PHw_(?AoISEXIX4hofVRiarEpht9|872"
        "x|x_fIGY!c!8$yHPp%T_(vQbP&A2%WRX#zNqIA#ODSJ<NgvPJgzkA5CL&7IyZb!?}&HOvO>ymIFP?d!{hFnvf+w&lnUjTtzMoi?^"
        "`;|WClk>0c{eUq>d(ku$FT{?l^zqA?1dQkSQ=EQc00BRfkVK&SflmE_@rZ#z`rsi&wzoU_NCmg1?xLGqAYt2O&0e5l$q4*XA0{`>"
        "9WN^ZLzxQFDP=r+xjW@l2|yfd0*p-_#;#T5jSP$Zo(g}KyDI!OrquM4rKs3AV)vq9qquFrdY*XCmH$6XeYb#7#;N_jn$7=%eE1m}"
        "EE`f8;b+Hp@90J4W!S$bBuQ=`zm3(5N*_yajI}vwa+-Yn=gsIH4jcVKPRMLF{r*f@I^CJ*oI42d^Dt}{5|J7aQlnCjyj5;Ww=#D4"
        "vMn0*OST15zK<(G7^;z@e^^Ia_Gh1K-Pf1BxM#N4RU}q7>W{lW){jE*_j1qY>}Q{Ph5v7t$FL;E|Dw=XBzf0Y;CCJN3mem#>8qx*"
        "6@TZCtAlAz+tGM1Ce6xQdqHQf-l03|3k9UT`Oj%HGv8C6i~&;C+{piZqM!I_uVY4p^5W~cOi&L8o3pCZ3*@L;GWh7O*7a0$_oNTq"
        "Y~MW@x8s(iIcR0yM{)3dDk<=NOriU}d#!CNCe&ZQ`ZrKZ0|XQR000O8001EXSzqrG0}KEFOeg>VBme*ab8=%ZVrgb?V_|G(Wo>O?"
        "FJxhKVJ~lHW^8G0WnXh<bS`jty&B7s<F@g>Rr?PJ4k>b1;>>LPNMy^0xa>+Ut}Vxvi&n*gL})}0MQ{Of9+}d=r@H}=07YrWb}G6^"
        "L<9YPp()P!$4*Puh*Ha9zAG!qWKp(deajjx8=*y8s+yU-065z`v5ji?%(S8|M5SuU{`KQ8tX8euD0OEt&iUInZ{L)MM(LIn<!)D%"
        "oBnw(%szbn)u_6E(121MhDZ7K%}%QW+p4OPTjatp2m4-ibt|>o(1;eXdLYX4QJM1bBdjAcZ{HMh=cUTE+$vopT0AU(zim5X?`F)X"
        "PH*J`<kBgde@B4D+cyk8<qkM|$ZlnuaC<gypBl+iwp=n;X@FF#^ivLOZ1?&74R;6eqh;IangzQ~*+}j7joc|MXH4usGPWFus;bDh"
        "3goR@!_(OSsO3WdaNO1)LPy#PaRkf2IEZ8gCJ~UMMr`kPEoZVaayVtW&99*O9`_Cn%}RsVM7@<sfOcker?lNO05bShHc1+4MU+71"
        "-=gZ|4_Ygo@Iz?~Y9-?H996HWh~NMw1;$W&0-xM3kejw$WJ(^I_9<67@9OPd)VC7tP)N@hy4~1L7(eG3#}fl%ksa+p_NOJGy~tKq"
        "o324m2Y$4y#X;txDDpy9vh9g)MXl;`D=OeH3J79i;4m2|^+&SO<YHq~)wME7hYi3p&oU0i4faBYXidE5(<j=~Q22#_#q|1sOqd)4"
        "<uU0s5p5LpQ-busrI6}F3F7b!)CM+X2$*0kJ;4CB)F7e(p9k^iEG~K0+=0vf8rlF4d9_z%Ee)u=Doq=x*5$#htIJ&pNANiS58<Dg"
        "rYc*)lH5tz$ht7gUvwwaS-hRR_tm>GO`?Y_cU)XOBg02N+R`ewVRL2!k^!2BCI3{1Ft4vZD7F|CJB~84`TVwQnWxD-Zopu-qw*0k"
        "Dxc&Cv~E$oU%CxD`p!^BYxqx?$|zychWFrj^`}qlC&eJ3Xi2mTwllS3C|geyYh8||2>5grDTDg<;mfZQms<$gHHR<Ka|%>3^1<%y"
        "ZvI!ErrBOT7Uiu3EBk;`hXaVZ#PCk~(6+0b;=Lz(nySxmGm%AK!AV&_K|nv#rx*wACawa!_ng1C3GIELp7$p_p3ry3Kjp$zzUP4="
        "_jxcc3%&^044jGIYSlFWwM>POo-Kg(zh@)#A+Jbdv=rlOEvhzJwhjOQS7iba`2s*}ftjWp6ZKXLH$&<I*Rb7!rN50Buyz56{Sktk"
        "H%)1WIt;#uz0WR^^Kl+AQI%U+8#zAUm+g_RzN^(k-Jd)T&6I5o<*1u6h{^vB`YP&K@~Q-$mp9k&Nu%D%LfE(L!^NEYlY|BoLb{T|"
        "w5)wlc$PV8nRi98-O2{iGSHS1x)O+duomXz)f(5pL1(^y^uCH&sH_Y5NQ`1Gl67~G(E7?yqOc?Y=X8R7w;V)R#2Is=rMMfGTKzQI"
        "P`2^_jGeZGUU5tN71S{Kh~fo$VZtzpKSsEn+$*#%!P7P7(wle}wvT1q$yhL8YZb)X8a0C2JVSyP9n_&MXe+ltliNsn9YI8#K?9SV"
        "kPXE4R?S3%RgU(?iDs}n_U?(}Egf8zYUQcJhRzBOw!HZoFv$CSrv-)Z@;bYjf$c<DI}iQ<4s7@I&w)Q<%XbaF9O8h-8;qz!StkLe"
        "07;5lE_(kz>w8g^g%#t<h(l9h;e)e-?^E(bS2xs|B+pj5mCLGYa&)_fCUvqOhi<Tvbz+gEzatzAXnM+e+xm*)*g>L#?PXQWRo7Yp"
        "dY0{_FZ-Fk60|@HwW@=5NqSCveT9$((NKpWn{x){If%yuzF-M_BZR(-?2x7o87>aWf}CI<Rtuuy9gj?fi9=h(1v;b5HR#n|c>=WS"
        ")gmzTx~KXoY-3Elm#iZ%v+TJW4(z#Ttsn3%V~<=RAF&mLF%r6GL<CMV@)zWX4j$Q`*(4MY9^+`F{p8=L7x06CE*Lw7QCf9JSA!VO"
        "aqf8cZnS`(JNdK#!#}Lx$NChEJOTu{OirQri7xfCGka_vhOBW9*Ev<6-&c#q>{XlFe(UT$y0$;UHt9wiQ#;esB{02{21Jg*$Jzfy"
        "TVWG-PCDx#=b%G|KeywglRJ;9Jr8t^)Oiz7Gj$GZd#7_y-zdFowa_fb;U~q8dI*w$g{^~v)rw6e^Bu&D)Qv9dHo3=$9>vSq4j+eW"
        "S<xLDW6#Z~EaYHcdr8ECNAo?U%`)LLKEoS2bVndrheh0M%X0anfE(89+J4XY<!P7Wo^$@a0zwDLR9!)8qPBRx(^Ae|bl5C{lSui9"
        "cWp!WSk~?(k?>8ZwnIlm8qMmIpQv!c!!OTzP(?KAbiC$UQA0k=B{qt==Omcz3{x`aXV_zBm3qLw673pZeLi<u1Ewo2&HQudHN5T`"
        "U0ne&XlaVrANi)dMXg`$JII;HA&%*y9kf_=<_Z<^S$Np-qO2P*=Kdh`UGCD_Wl_T!pR>%Yzvm|*p{ZfxY+{4MG7XIGM}9{<QWI5`"
        "1Y@;FH&cTif<5{x7e&}3WwHwrq^}!G1|IFz191r*jhS+YiR8|nkEsRT;+U{AvcNg4A#o_gF7gr3_sAOsIp`@zyE%<v=y-N~9SY}@"
        "C$$M3Hqxt>*{-Xq18OGGe04qlo0#v`&tH9Ugx&u9Z&q1Pm~we==klc{E-HQf7;<!<j#&_G<C;9Wo{eU&UJXDqjfcYkp>jC7-MmV)"
        "Myq>SBY}}B$-AOCWiSupuOBXko)-ni%cHE`2l`=_b$Or5Cuc{0WF|F*@N^9H?j2rmPX{p*PR=I>Gd_YRCNrGJi2t_}qdz({*>*Tl"
        "C?IrIl{+=j#Mya+T{=~{Qzg0)jjJ*RcVI{g8z2d{|2^crD$E~tq795-aFsacsKjAYh=tplX(@{2=K8}Ia6iH?2#5`joN{0%>zV@R"
        "AcQbBs@pKx15`>t2@M2JiV_MdS~utb{-J8R970|HkE|IcpE+)}*OPD9z0~FIiAnf=q8Tpbcz1(!qgtPkhc009rJ#EkNbD<0jPy2S"
        ";L5`M<pki=KcO5ZN$AKG;~tzL5XwDjGDmMrC<cIyWj~qi^Fe=z+y$711nWAJO}SHBQQgYJK}6~rzel{7B%*^JzthFgJ>um#=45~c"
        "*=wThe7;+-dz-Om@B_T6M}&hIlh7;0JCZ0wdL$u(m5UNZQg{M6oRYV?Y@hmxfqKAO`?G7Np%;Lv;qUQ5aUmD7<;C`YP%_{?8o4>X"
        "Jj0W;71AV`wc=47nTlHH4tShI^xUpEI;r7nQZRvEtSYhb;P@c&0+Tjj8IVvGh#0Wx;KctXQv=C_Cc{BQ;NbYWB#qw2*rWYx7nkbS"
        "k!q;K5;i!v(yOSB2lD~ivyK(zx@<5VU$4jjFBjlQRd$|d-y#g&;jQoPRz!I${|=!x1O`ktAgw3Z^Dkc`Udbt<z}Lu*&J;Ie)dB0y"
        "v41<cVH!K7oFHas>Uv6ilEL5k|A(IC^H)OkFd`_cpv|8cfdoCB<LfRBZU8jhI5*pPX|mz3E$8n2t_e8@wtdJsXw<;{&Z)&K22mq*"
        "jx!K`UP`{M(T%S-bP$rSi&n@W6kSnEfbdx#<wH9_vukw8h>jg!Cw@B(;cYLehY5M`7<3LlOZ;aF^@0#ySAE$opkZjotpNr!<~jU%"
        "0CW|5kmc>ZHJNV-{x0CMM(^$qox%Sp-Ail}4&*gBa4=XY{9FRv%>H4@jV;)wE8r`$$iBxviy$pHXr$-SfQ~+QaL7QzgKwyx`~ahv"
        "(Yqvku>tW4>zLMo0;nkK1@7xJ{)J&M2!^5?3;F9LmU?%!xnbWf8A0*K&0lV=zy1*CNejfNxTtt{3xCxc3KSDe;Ur+Kx|8+zO@(}H"
        "z<jV}8I4IC571%9+>D!g#~kd7*o>|V&Z5EI2q#q5=qx;cdGldt+VM+l+m8N06ZcJ+%yb)G>Pbcuu2^}4w#wmxI^t!&q|7*E`Ny#$"
        "=^rIhFF@I*MOVi)2f}|jb!AmpHw&GK@fhz8!pPg|*|T;0#<%0omgi#@92?&Nwryf}0pPz-O9KQH000080000X0KrM{K}!b!0QM6A"
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
        "qo9Y?>p~=*$(1#26FuMN<~tz{F0TAbx8wfK(^G#;^xzswsUa8oT*5}~%wV3wu|)z$3}>F^lwB**)ZxKjP)h>@6aWAK2mk;8Apkp+"
        "+D~2$008YO001KZ0047xV=rQ9W^ZF*Y-eR{ZDB8DVRT_Ha%FIDb7^#Ma(OOrdEHuBliRoves`7s1H-&fyAns5(^g7bo7#9>TS@KV"
        "@opX_6Ba~*BW5U)20&?KXa0M-8vqF&b8PKCg;n;5K%>#<<3l$<-}isYih?JeENc;SPsSU*Wzu6s;=Rw5Cks}|O{u&sQ(R;$&ptDi"
        "m4%m<!Yk7>&kF7dUX|X}<;T$X{nxKvzsj~%DU`Qia+7E4=KVjiESeX=o1d~?SE7u$l&@c<qTG6MnddylX=KROv1jPM2ML%eg`rBA"
        "VsXx-<kC_!O@^dm3MH`JKfx=?M}6xoEn}A7^6i#|aakzAV%2c^z$=!CclT^t<$Q{-rOaUN!)4ZezcLCM<BH-CYksdR<Cqm?kpW3$"
        "RU4t6rk>=u=14T0BQx?*&agVLRwHuo`jrO{tE&7EQD3CWW|pb$z0Fj_i(MwlV#^ELs;N<1(zHmHa*~Bwp>cwWyZZ5$58u6uKD_(&"
        "`>XGN{Nc)5d0#ZWpJw-}7TjNYf$xfXBoBq!aFxZgc4;$9J#7Zgb_;w1D`bT~nRdBGe8=+N%bIvJoaLTJ(9Z?3)os+CfzCydZ;!72"
        "`EGWJ1hbDQ(`>Pr|J|QDlyCV-x;QyMU-RSmEnBeTlX-G{da^#9&)=M!ot{G1o3pd|qBBUZ&$M)e5q!-Hu)^-RCsD!ISuIH(Lo@sJ"
        "xWjZYKYeqO@OgSRPx)EG&Q8zXo*rM!&)CIze7*n~E*31kNM9g{;PMkG`sZ2*aFC-9@2+M)qx#}Ge{(ipFXFfJWC1Vhi{xZ|oSwki"
        "**TBrZ{y>5ewwn5svpp6UcdUb3CsX|`7<w8A4Sc<rKrt`rjJbCEww0FAr&iP9%aeWlS&xBB$ygBfcAEo1i)~5M47vU=pxSRnn?~3"
        "w%umRY0adp-9d_pTds68In8+%^FnqAxm1yiMOJBc-haIxN`S@{{V`E?P@A%F!|zd2!9UfR;7B7MCCXWGTeDj&2K4xP0$=Mq;VG>d"
        "^aorrbqE>_Og+GHSU<2%vRGYH6a076on_+9zQJZ=5U^>DUC1T`*~%TS1cw+3{EbqTTpk?(x@_t-K;HJqpBNfhr$iG%Co}@TtO|VW"
        "&&xd*!Nh1j^@<X=g)pu2e2WSXgn!LupXal;Q8>H#`pA6&FC%#-9_&MY82aEZx@Fn=IC!|^-mlpDPKZ(jeqG4ALbT*bq=hAr^NkQj"
        "EwD)Y3%GxIFrLPF0X>o{EAxH>n}N{_OmCIi3IL00dU3eZH7*|+jQB+HA<nrK%+4ROBnkAw+YW0OGygTc-*nacdsO_a)L8+9SrWC<"
        "Te$@X05*o{x~19U6a;*P`&%T+ec(j*6z17$|0F*g$B5MCEx0gHa({z*#u@I4a$mSxuI6T?N7MpyLj-{|24JRyE1>QERM&d!jRlej"
        "p%nc^3%FVtaBa~DV~OoU02pKVTI4>i$qwc@NbI`~O8DfCLzEOX<w_$%A+1{Hd>a9hDgqPBhXBpxBg?!-CK~F-SW&wfu5r^H@4#sz"
        "E8$L>anTVt8DZ{Vz2C(HmgfQJ-Sh`L!l?<Rr#pis6Us$*XnumYsn*g|FLG*xojUSNCQtgs7RG1AZ8UJxjOR(?+L(Ak*5wFiCqfCI"
        "1T}~-HcS6HyS({wF+H6>_*TfL<+5QmqU)hBS`&@3oF1c{u`Z*rb^%W>a=^LKu*<_qF7*UmZ0H8IllVB?+D0cN9aY)bbebSYI~4tm"
        "aN_jpHCe!PI!IO4&75|vCkL992pSJJ$$?K>Q@=GT*w%De{PA7R4ZV|406934hMkp3u5{qwF8$cxHEeNCs`1;IxH(9q?qANKs|}N+"
        "m^GUeWztjj#N8ZN64|V-+>_62Yz~AK9Bl`-7QDs{;%3=+8iE=U(Wo{G@h#Mu2niozEeaJyEA$?+0Zmp`ya*u8E)xKOm0zoL_6DSf"
        "qId)Tp7Y*zh~Pt2FyPN9fe`^|tSgOB;B!ELp)wiN!x9<b7_$hTKeh~a6&N0%0fz+*A)R6^Ib7E;KX3vX*-b%msh&>)jLC57$heM-"
        "m=NsXCNX-6P5u7V+A1bK)m*vDGnH<20KVnkQlM$7Pv}3cP1dx<fl;bY;ZK?nYj^D!VFvyat#tt%(BDXApZRKW{HDKDM`R)lp{lM9"
        "sFa|wjeL39Y4ntWED^{Fn}Tn=Ul&8~GiA>duE4=9W8e3G1Si{O$kD4{Q!Pbyn-whge);eZ*xk0<S~1w|y>(u$<zK**9`0JYj>>aN"
        "Wf2#oJ*6Rkfq;vmtjJUp1(N3}rZ=T>!)@w)h4li&YgBNxFU8%`TbE@HZg$CF;jX%nLx>D)6yk3z1R%ZG@nF)U>~N(7)_J!gev8RV"
        "7H>HKN8(5g>AQgjBPEn-x>h29SRYg;6$csW9OY_Hn?g|+LD$yTK2|AE!OmvzW0qvXr>#tTjhW<T+8qQayXu{*bao4TVph>XMp>(>"
        "R#g7>C{(uP&<h?ocY3<vsTUFIof87N4>beN8{&-L96O=NDFZiz`r7xtw#=nUxDa8&k$+%dVoGUf&M-H=UY^Y7H(kyXSJAYsx7=V+"
        "uDjX!gRV17$y=Abbby?1%kC*$z`9w1COn(Z=ek*2xoM7|1OJ&#rCT@HsB-bdWCW(FZ8HRvC8RW=B{reAa=iS=3(O&BC^zi*>^!g>"
        "LIPTF`AN9p_eplk5iUT>rVbvUuIAo`N=O-U4z9oBkt%~`(&v#_v!aYE;u1WMKRzG72ON`w9mx<4qjY5F(gGm1{!psRvMv%M$5nH7"
        "tTkuyYJesa@hJy5g%;RypwN?uUP<W6YMp}?9&Ya-9w8F(TuSN?@<{%uyt4;N9l>4{^7u4iALo$D==j${!|!zLc;kyT^h2nJB1q^V"
        "lRiG+^Ae1}=XspK$6^Hp;`Gr#G<JLrGg=cqi5vK|*g<;w!_cEAxDtn+j7_~cA{hAx?FL%?G}&$jZX%5(cU)xY!Pf{AI@J?~<G;V?"
        "-w%Di!^f^;tjnZwkE);F$GlRGQh#rVjf6#a*%VH=WStp|EO_#Npw<Ei=!kAa!d%Xj;95SBe{SpQP*yn*uKj<_U4dv<0an+JL#h(e"
        "C1{&>(@C87fa#<WVBDHSHDW-#%E^klXG6$om`<^=Dug*{cq=R7%A%)=Ehuj4-DO2G^-4UL!n75JT3%-K5C?FG_otP&_Ax>9F&#u+"
        "_T=o#+F#7)=clJIC;hxQJM$l2f)y<)K)G@!BiMQXaDjIX%8h`1)Fi)sXF&a%rFSo!0&MM}$fP$|8|v#7?*0L4VYp^Z@=cp6;HqKD"
        "fDAwJzF9fR#cx_F?$eaVuj>SF2r$}tGF&0u;Zb%n;;u1MGn^a!m5f3rZK9*U-ciq@yTz5Ar?0YJqN*Y3mm))=zY*yeG~x4S?>l|{"
        "cyx97(W~(+oTh@Z`2agO<t4z7$8>B6TRKmq$1vWioGZK&2F*it72m-|6}~hgz-Abt=#V(uizA6F7_tgV_ax|E)&|zero?hGOjU;|"
        "Vw?Z9$A_w}3^O@#jw>plf{esHEEJ$vkr+%J&1!X*b*ANr1cpSFcf0`Cwnt!JnZMvI@g3JE?pXA2y2P=G$BS-v3@!S<xoD1PwkDlE"
        "oxeTDi(ky5Y4LQ{G9{*l)lFNR(j_V8a{QiZbUYf|WC>(R9-b7{`l|t1@z?`(d3%v5a9EtG1o1(I44g)rk?K?94cZWR=45foUR;_J"
        "sD~FFnf8xzk)w0c+)e(iU339}nE6<7@Z{VMR={XTez)!sy50WmT+aS!&&FUcWI{<inF<Cw`OBu(#DmG3j}RU&e40DLNyp>x#(#L2"
        "+7i=VNtYX7tB#jGjMx;<d9*pnUs|!+!B4J}Zo{`s+_?t+RD&}iNHi8+bC8PDcNPeh<6mGjW^e8|C@AcQ8R)&M$3N0=t63YudZFgH"
        "y^UlFIUQ=NVe~cUp&EqWmwDo`H0Ahh3i+}QBvhBaoF+tu41z;GW`%BLHpq)#I-u<y`8Xu#$?RTxcTVw*3VTKGV2rJPo1NP&f;B%;"
        "?%S_N2tzb%M(R-ES0uipV?Fn-`z3{YXS6h`Zkt?6bE#*D^xLq%`GlZL7>!&m_yixz){))5=D^$Z+xNcg_SSCcx`{($ObPwg6J3$5"
        "=V}W8k>leo;jJj^3VB$k?@mE?NXt<a>Z})S@fa6--(lCQUq_Y_HC~BR+TSi=>)_}C`ryx4UwbZQlWAShKnP$u&$M3gG{CqxQtq#P"
        "*IoS^aK(WW_Xbc1MSaEJJOL;)V)DeiE|WIChDT}Te(HSoVRqGS@cePnK-bOIw-MW&KKJJFd9Qi7lLE*YODu<W=zO%~+=^M!w#lU&"
        "er#h0412oEZ*RR?HMDJAXLBv2?b2q3Tz@=hzHZPtArR`UZ2ENehsT~&$h8OBoi7sG?;oDs%_x6C&l|Yz%TfsBHb@}a%(;~*=y|&9"
        "lJ1e17Djua+fM6s-@B=Bhpz?C9$$0&>4vs*!}vc?O9KQH000080000X0PCT}PqrBV0I6I604e|g0CRF<FJftCZ)0I>XJu_|VJ~E1"
        "bYU-YWpHnEX>@OLd0%61a&UEXE^v9xTy1mQHj@6Hs{98o?>=zs8A@`Tt*o-+J+D39Rjp&YBsrCotYvXX%nTzA$?!$e#Owcl-HjIl"
        "AUGN&byYXJvG@WsfJUSH=|+RT?|;aOf~8)|?pyC4S?k4hvdL~)<Hbel<ypdthNV+4YnqPD$|B!+HLJ=dYs-4)UA(#U;&clyv$6<%"
        "-~alnufNK+Rapa}rd8LQxY^{{vcGTJDopZ>6@8gsnzB$gHB*;uT(7dczwhcCNQJDf%i4ZevtK)g?XK!_>m_BLv!sQtHJ&K<s_crE"
        ")uKQew{eokO~X_H7+fAFOcbc%7W*w<et_Gl_o1uV2jIXx-r|6%=rUWCNu00Qb{mIDS+sSWw2JJx;@KM*%T&9+DB`Nwl&y;l_AgAj"
        "I>s?bQo+|>dGPzNIRD}7@^bMydiDPGB6@rFYVq!3aWUm@UzI6)f6MATuFi^--HXQwJdT=dUBoR6LOf*mNH?m=G8a#f-%*;aftBJ("
        "Q{`D3!LSNoC`+T7C1ss9@OCoV@p84wvjW;%wX*Md8K)5q2xw4)Uem@)Xl<1hYgSivR*Z83W+U13Qx#WLzKiH#)Rs{r7&-O$Bb>Vi"
        "=*KG-p~%vMQQDeNn|!X|O<Z#lynq<I;7rY6NKM(*36v@C8ub`j%u*2WtleqP1c^3oZX)ao*5O>l|B*0yFZmLbl5qhw*eNFgOY9sP"
        "#FhHbZb|crJoRE=2i$G9S^K?LFqnCmRuVz4#7mZREz?&n)|3kt38!r3MNwA3EJtGGH18~5P5axrEL-jNiFO_5d3hHVti3DioB5kK"
        "Z<zLW8{bDlvqZ~X%bNMg@o{v_|Fg=;i4+CD9V>@Yim%m+$*I}#5WDNmY1syNJ%MR%%KVlE6RV`rd#GaETV)4hf%1cAjymSJL?!c~"
        "qZW-)6sBu^#$@e)=E*~;(wQLT52_pT7+^sXd5sTzy+cu<tixs#Kl}F2L7=ES_5A<w;a{q~j2jkliJOG1KzRx#lW@cCIU$|JxblE-"
        "KPN|CAesZ3A!zt;{hO~<$W~qf#jsw;?z5(C0)1Rsk>TuF<w$JftYIE%-gjmD224P@K<g1m<0siDjUQM$kBnNe4)tysf)#FW(yR_p"
        "ITUR(zpOjZ>(tq(yrEmGX3J2q#5GO_jqEOKH&N5AR@r^vgLbRiEY9urR<M5K7VCl*Enq01CxdTnZV>%?=Xi=gY{)W^A>5V~D*|^7"
        "YS~9K-JBjBNi`F$fl#*u^R_+uEXCr>5ud-K&$e8A@u%a|+ic6quAM*scBIe*5Cn*AQ-S$rBLgEsWgF+f(&O<WG=9bYkjCxbEq!WY"
        "13)L=*8ox2fbe+9rYml|)4A$ah}u6mIezxsd*b0A*Sxqp{B-W+&<!FvdAJgYJFoX*L<t?xF#h;b_%Y5qCf0zYl9qP`+2>`Ku2~zg"
        "`-HKSrM5u2ew>HlU7fXT?~s<ju$$p4Y^zcxcR;;Ii{U>nFF!1(G(e&e$<YyF`0@NL4RoaJB^SeeMP6~#1MQ8KA~~YwMrHwKPp;wl"
        "+<UfXu7>lYEQFdMu<J!bm~9njIRH5jWP&vU6YKPPsTasLK-FWaT2e-U!$4=lAHavdu|vh|kBL(c*^&|<OT1=esc?vZUSz|TRA6td"
        "Ik0e+8oQ;Btk*a;4lzbdw4*%79Hw#BifLn$BMQ3;04570U$|ABYY>~y$ka3Okz-Oc>NNuZUpN@ZaU^1cAs>lzgaq(kG6Mj^1#bY%"
        "X&li&mxZ;YhLgDD>YCqMYPeynAd|0vKy#B-6&TyObi(8dfzsZU1(RM1_Q-g|H8}SKx&G#Z@$3H}a=`T`MyUxXn$9iTE&&XK<s81}"
        "(p0xN48y4lc&)IAw`@X&xv>OJ;uEZ_TA}hdXYdp|9Qa4jHF)iZp^sgl@}u~~wTf_*5o?mTVzwTJJ`m>st013TWI1E0)vM#{VFATx"
        "!ZS5Fd4WIkr{2lrYIbrR0<7xi$-#h{g{v;lx9D&M{?+m9FY#=3{rUMZn0`T>7=UH@t<D~|7PKXEyc1)_cATA@qQb!rmP%%Mf8tzu"
        ")NF0Lyv?c{9l^3~9MpxJWDO!mSNAb+kQE(sikqs9tH4=Ztg$N}9PgVZD(k2yi<+&wJg)s~#2hD&Xfvl}2K<<Ke{~wIJ)Gd^8C>N&"
        "ZlxgG3{gNx3=DPLpc4}HDBXneggqk#H1S?&uF*kVR5~7gL2}z406cOGuufn>hrg7WV9V9X>9sv1Ck8iT5P>9U)fFJ9@9Vw^?uw?X"
        "5CdQ!1$q1E*_FKr6R?o0GEajE>L^s)kK<#)Ei7QJ6sMfxE<>-06=9sFfumKsXbYBWi#|A)rBDPiNa(qwST!(!SGTev=tEpAz3tFB"
        "0qY*E5|!lOU6Va<Mo)%omQXscIzg?VRe0Qzpn7D;ae0u=9IObZ_)u(iF8d#*jGS%1Yge=HYzjh%J{B1kdQC+i6eds-5m5W?@j~Q("
        "Iws48UEJ5gGAjUsK9WAI=`aQgMH-k9R3%4KYFf&WP?W;13BzNnhT#+;FyC>kB&AyNRRFDFyf)xCN}{K>1HCVF5@5N{eRF<Q=2^1y"
        "Pdy*y_y!j2$ICb0MM5p}W-+iHA==B5=;F<#?l}8YfG7ivUh^461YuH;P>PhMUO-F)yh^jA4aBWMW7QSp<ivTDY*=ywUJ=?$7*@-m"
        "o*vQrqNB{DTo)+QVy_^e`8btf<Mf^&eb_Ku2SM#$e0crS>@6=lJ0nB9%2<8+sd*Ay{rgjK{dD4sNmX--WV8vq#i_$(43EQgU3OJ)"
        "f*`rhDges^xFDvG*8*zdq8JOIdkjYEfrh;)z=!cUA)*RxV#mxTfMTG}Hmk*c2x_7YtJ!krt-)h!aT{e-Gv<{mFWJQPn$glBVp<%v"
        "NMSxV%xlV3dl%OfjH>GLHbc{7RnRej7^V;6=&PRjaTUXC%8i`9K5qwn2I}dv@VI~FpvkGsu$XhuHA^$(|FWZ<3Z&gN#A9w^=?T{X"
        "h|0Iko{__Ff9KFGK<O`=boR|G$z$k?KTutPUob$SzzXCxLU)Zj2--pGlr@Y3uAvj#vegRMT9+gS+j}-_??E|fOXL>nl4=L;R+(li"
        "_zUbff~FN?nP>23l!IB593m!ItQQj^4gV+kr0FYIk#uOWFg_~h1GFv!yPX1k=rr=J>Xsl|URrMBtk^3<T_J_g(5%`z>vakR=~Jar"
        "aB`;7{!O_;mt*pM|4oVUU+6Y&WkH<`Bc(HKv*Y~YQmz1aljL2>rXJtL0mR~ExKAfs0=_3gY~kBLNaVf}JAL?k=-Hy58TG;0?~D-q"
        "ibQJ!Sq)8*=w>9k86~?UH(Wo3q{und*(?-sN($H3$`ZjCJ=5v{uu(H%cal#OpM<{3)ZQqN_~e1j0E~qdJ8P|O<NAiX>5_`m>m!d|"
        "6?Hg}KCINL&RC0R!x5?T*G1>8S;1;>DunDxu^cN7!Up*Az*<^|Z3oSYeTo@wxP4q@MH=Th7|4Hr3O|Kj@V8S@;6LD$W5eQhatfyM"
        "Y4GXd=>$yXfBV|H05U(Aj7GE|+ad6JvcFFWEd30_RTwsqf*%&3vB_}(O%#}c5)~mqV11;<1P-(g>n;CBwPJR*ZyZq@%&*I*vpQ^O"
        "2E@pchR_mM>cO?4b$C4`@sE{dsj2u*vs|+b4ONk_z}kwoLuQ{TiiNB=rDvh+KZ-X#AZ;ONgjDMbryyI<;%-={MKICw2$S0S4ve<("
        "2P1`w992X4fL8-HSX*zS7>)EhP_gDj@tER)iYyeIXhA2mk$Wpz0V*L>c4&@MmojHo;kz`r_2%$J(Agvi=lg^yTe){P!I-=s`rL}^"
        "&!3{hKctk$J0f)D4nLXZYGfk>r_B^zDexgI_if{fcZun-ql|oDBjpJa9QkrdVcje8A$TS4)Daex^&ExAgK^ZkLU(-ZpT9@Z+GWD_"
        "zCEbiHa)0V-5f-2hT!>-avTsJO7hUr2(k|rN1zD6<&ZM_Xnh$zc1Q!75fu25V5_l)8wyATRQ_>0fX)>;(&PvUr7-x#Ktssjfa>sr"
        "U`$3aMVyebr=$>%L9eUB(5qVk^nJTF`i+1|0$8z<$DCIJqf-e_rUIC%9@?P95!V|IHi9-I%=2qoGdwEokqpw9v>m#@=jxk>EKh~u"
        "ase;VG6pk7)a++p5D~UqPRwaPcAr~wih7{wb_hKF`l(PgMRtn8I>&0Udalto!$*=&dX`roQu;tCW|ZXjmZfX8)W8*a-PM>L=Yh94"
        "E4wx+!Ic2FrRo}wf;3eLo>=oe_+N~NJsV6GLm!gmm|EA4CwcLdA-;_U&P)q}-WZ+>_kWsS5r2LoN#IYU^N=~Vc9RJ&iuL!H{21C%"
        "O_!)bczdD9HMAQL1w)b27Wl8u7cVat-sQ``zg>7|Z@hQ!FTKT2XBU?jYE*&Bc=Rq8KV44zJ@l*)!XD~kI+~zdt)M0E?A_(!AB%JE"
        "!}-|{FVBDW{<ZjdTE#m`&*RMwq=P!3`ZjmuLI!GPv>G_Ysm!O+d2z9LyLfe}k$iLh{s+Yde-f@3K*F+MmOmyaqRr=*VgpPOdQfTf"
        "!YmRddCID68sl9srZjXUdGQvl76wB*1LbypiA;XqQ~mMf+mE2~g1=$Nu%G4W#4*7<P|Lx)Tnm`5N$WtDd(;MeN^y1Y`6KUrub>Fl"
        "i;A)y@m~sx_x}7f$j9G*7NvA~u!g_OSe~k0U}kiP^bpe%VPh0ce9OdYw)#Cth#cdn>Z0Q4l=IZ+LfUPsz6*ykImye04Y;go&|A%="
        "T6S5U>e;qj-3fkk<0SKGVfrXAF4LAVvyTKX0?i{Hw&e{&^<75^Vk?=?+;%CFE>_uEJOUn6owP-IS()W3Ojm;03z)vD@=ifHbMH3K"
        "VD$5E<#RIZ_pNyR{P-__o=Uc$b*;sVXUEUKlP|L7Ms7#0vnA9KFUGTLtGZkR<Y?x&;f0yRT3oDV>^J7cp4{o2Es;a8(m7AR_b^d-"
        "hkHhOCnMIe2#i4~Ar*Z{%$ev89F(?V@e`nuvmX}kE~ASNi&qz8wJ76SOpKA+P6<QMCv-Q{NKM@sofZ^Pfgo2C7cJXd)*t3>OZi0I"
        "#{<*?RekoGJ5<2;to9K<++pP<nC?w<`^^8ccNYl=GYJ)4fC=~ivm(&NXllvM)RdX2DJfH1K5$?DWN)9c;x^8+R4gJ<RdxtqV!vd-"
        "o(!_{19L?wOtZ$AG^Vlk@2`9uMcw+>S3dCz!*tQFU5q;)S;2e(XqS13J5?fpDoH^d{=A>646vWvU2pPk-3+QEb=fq?9tLC7=a`DC"
        "$}s-8;ogb1=y`u5@ly7V>i~&u_+3AH{N9z2+$R_>=%uoXs}3CqjPPTA7Uf5pS@<|D$|{o9k*(8>{^DRj^<N&I5#Wb)FjoqD0G<Uo"
        "KRJu}hDV&^_V;ckxA;<B7BW0D5{iN2HDzs?M@sZl0x+0(&67Ol)OkyA`Us&fkF!KXmo-O9TAGk!KiyrWBsLrfA1;3(rGlXD+<9Dr"
        "a}V2-=b2zRn@ZHUlac(aZZI)qg^(Bo1?6!CP1uSJ@R0}&*3;q36ZKL^AkR-w#Pt*f4!Zib2H6X$nMkhCp?(ase)AxiY4oJ2E^)gn"
        "I*4N5rTrv@g4$2a#XePWA9Yfa%`jy!PB6=X``cpaDkG-H5^0T(5pkp_N?<uRHt@Ml2QSh<AKEJdA(3)6_1sQ{QOv#eetBqMAFFB#"
        "Rez&<n!av9loKrsidM`tEl|+1HCj=_>Oe`;3TXumdxU<qVW0GZrllem+<2lFR!0aUu_2Ag5}HDnDKub$`wRs9lasp))3wK$*7TH8"
        "3cOz9<3W^tP8S_d5n5@|yUB(c0%sl+F!urH-t0tT(cX)YjyrCSQo|uR9=ZWBiSP@p-z33TKD&<rK>(t#^JutnBCv*meN6vjppbpr"
        "uTYzZqV$z2d2KpoKF3Ks;Lpnrb_KBjc2<%SRjd%AT&gle#7hlQ?9@8F;swCG$3<CWi8Sp1ERL^T`9qR9vY`QU92*b4vXI?UqfoDW"
        "JGW0ZRI^{`5acPIg>u!yB5{LPp-8Dc^(R*EG1XKccYe}V6Mz?U?^x@wEEjzC4ItZeTPCVt(5oi^z*RAWXAH^P$rl9?E=cStN8UrY"
        "U!&<sS)MH!0zf+F%l&}w!6E^0h_fwbt^l4(@nmdSb-8fzV?Ypy!iSPy$9E_}tbnc})xbV2sp&^+*Ie-8uXBB5_WsczZA|QWs_`N("
        "b^Z9j_UA+EOs-_G4_d4*Q#)ilw=jTV^L3aNpj`?WYnJ-bJJ<c@VZI;D$9(jtXdp}wa@H14xn9Ok`%NVn9%eaGnLIAm9iGBa-cYq`"
        "Hzn?m!j<9v$luWAz+^&EZP-lTNgywM0T3AVTpd&KkFeF5&y!LS(I3=$?|TprGzY|yKCfvV)q!gi8Pif8)Ud1~X6@ImY~w+j{dr7>"
        ")d^G|ro*Ev{7D}@Pd0YmEDT7|e@q^4+F{NNEY66|_(yV2TUd}010tKU-Vf~xj8&GkNVyLG5T*~+u~s_JYF4v1+xcP2{k!l(-q|*)"
        "?ZD2+J%@fHYskOPxI25LHXE)<-NEo`V)~$S_!et$El6&w+S$0awj=CZV5JFYJ>ad{pqTsK%XhCmk&JG2-xdr!VNURT^eDSyoYdU#"
        "vEu3Qm2@kHZw%jlF76!VgmRjTzm5`^=`XP<xeU#G45~UfieVi@EF&Kdc!a<#*$RU_eL{!K<}gz^M)Np~9vwu6O1f!aej@J3Sm__b"
        "F*uVVYWfrarBmsnBKsxISRwS)dTPEJJA{ME9tk73;o$^+N(dkBJ0-hMrPH2d$zu?%W2y2iBLF&a04oHuHK9I87HI<+>BaatYWx8h"
        "!DvQ<$=-_66;ikHqWsCjtrxqZjqmvxQf+|vF_5Gqw}<W@JWT55l8p4p87B)IHPq&w^PyFoQCjn8sMMhw_$Mpurw%9oQCdO<1V-zK"
        "aQVIkuh9{u#~f7r*|v+KBRK^H5JVc$5}64GW51H_GI-@4*uto(SdxJP78;2naYN(WlYe3Q*9Tc?qcgMyR(W}6F9Jji^vx9_EYFO{"
        "hQM7h*a7$7o;-6?_&qW)6oKc!A=;Pd<3#54m>u3bJ<x(YOY1N(q`KzT>^Mx!s%7=ucV)y-@KX1%BppXt%Qu~C@X=-exL#HJlXLSk"
        "JeEvN_L$vBLZ!L9a20kWa7aKhw;#9`Pz<JnwUNRa80yisN-#I4-z5>b?@2OS?q<)!V;8$bk8ap*J_w=7A4wZ7k2M&t`A8a>FBSl}"
        "wEzVIen2ski?arorvY1~R+@{IL<TEGLK20GOmw7h0#bs}5jj{S>b20V^myi-j8W&sbh=u_u}W$snB9_(U0riidTQ3wns%3W59EtW"
        ">iH%<k9P8HP(H$apu-;U0D4>Dq0;ti2k|93uY5NiK+;^f-w`dspB9{PC}$Oqar1Lu2IOE{+Z9wE)Z%@XXSev>m>f|BikN62pgurx"
        "cqX8p<>mZ3hlXIx7yn{A*W<zGcdYk^e6?fG*E;0K0WRa(em#n)c$iKEZK%K@AwOPgJhZ2=O>C<JwD-G;pE|`T4or!P)afB2W2QVA"
        "MLZQPei<RJVwmd5*b35Ol+Y)2r-h+;c$zXwtmw5FAGTzn>9#>-e?o*D6LKtun{ktHotMkNe}eHHSZ!g(qkx0cz%T551Mi9V?8%dF"
        "w6Fcx-h`aArY-T1W26yqu6Fxe#^XYWc5BAw%md8{cB>p1YMy*0;}NAfAac5$@(=OR*+=IG`TB12s(e%a|M`BFGhg6T<rz44ir~Ga"
        "Xb)18{Sv@}{f%8r#<ZjU@UB|G6fKYF<6SCFNa4OHF62bej0k3;Way|!H?2PR862Y_K<s$IN1<TQL*9@Lb@ur~d@_bO<NzmPJQ3Cy"
        "ElDj39{#lsFjFQ^ROPX2Zh^<i2DxEl73eRb4^%j=x3C<#t=Oo5_s{M3+@S!8?(#gMMxYeoO6n1!zi%eq(b2GG)4}B5f}2L)W6Fw1"
        "%_J(r`pP!|Ep7@4Sja<$bbyVwF(w)c^!GBDZMU7+x4~}$2z5#uH2v}I8LXllqriM8KH-D^9UI8@2QnVq$EEL|aOHtOP-DgrIszYF"
        "cqctCl;SdXcx3{0|BX^UGbDz$Q<=3V6rFxFYZ&Y_G70qzKO^<;k7W_cWI@ikxt&3#lQYuz{$O&Uo<n$Oa^W-8#CQs!{6Hf2{3w0c"
        "t3YRZxL`c~!2;!-9*I63$)gI^r-6yP7>C+)>oqL)RSdGV>D1?SH!HkTUk;N=dIx{8_2T`<^H+=LeDUMi#o7CJ7yl<&ejob}ujrXv"
        "yn(R)Y{ZJyVtc~w))g-+&+oyCpcBG#<9ImqX+|BKk+{!JPL6Lv%)-sJwBa$IT!Zf9oFl)EI31L%<5{_)ixZ4w{@STjWYKw%-)~Ka"
        ");l#iGsY)}V&rduiUa|e_f<}=jrSng?XH5ABEk;3qN25it8lJm#37E{tn^}hHPOFjhL}(u`%{Yh)vA%SJ~HRj3FCTUL^t5iP&N3o"
        "MmF>i_11JHF`~6CeKDHzW9pAF7;#uI)z&QICc;d?({lQ#>rxNNi+(o$7f?$B1QY-O00;m803iVX7ro5)2mk=48~^|z0001Ua$_%I"
        "X=ZO@VQgn*ZEayMWMOn+FLQTgb8TsKXfAMh%~@M--8K?_H{k!k)xJpUbL6&rLE++TcNa;z1s2^RaodN)J)kAZhmAz4Bo+HK$ba9N"
        "Ath0l94B2As2>uGm*H?O-+V(SN%G{Azpz>tyk><~&hWyqH@uc5cT%e_n9`1M_Tq=9TiZx?V5PoSwdSQTOdIBeb?m^664vJDXXj_*"
        "!j`+@z@%lj-Q8BIvMZ#z%h~k-hDuS(TVc2p^^qBnVPxS%35S^m&i)02AGj)OVb9OLyMDI)oHe?Xl`Kdp%XG!WgR~ALvP#0T4TI(E"
        "jWDoCz>#`Tpp-@we%C#!^_6s>Xp$tTt!!Ft9IUA-P-yr*aC?Zqf3jK)A4Uwn+U~YBx`4Y6Lryd;YIJ^98Qn0jA}?xgEhyC=)(?dx"
        "pe7#HPk#%as2a`P$_hPmCz^)my?X{p&qd3ndHTScwiX-w)K)rej?d)u`-S4IJ!pq=&(BIxF&jqqc-6H3x?0(jvF+a(IOyv9jKNRK"
        "kLZ22V=vPxN8LB9BGrU~wH3rf#iU{u6cptkIS*cXEsndIH@78c)fMvsvKud2R;(5(ecO_hcEk24C>;1MV}IDud>F`rLil62w1R!l"
        ">rOm1Mw>Kw(zP{4ix_T7cD8wv`EBS0xCDAOFT19-z1$5G%63NVxh<sJJ>#_nTUszRgsV1omnNHJ!;-5c%W|O#U5Yg6T($kY7j1;="
        "jOv3Zw+H^&U;mckfmwbK52d_=fKM~;jDHN_kb=j55o&jBIsub3hI#h_v+21gv?;@trLvBzLhNOEg{~QIWQE%s@kUyR_h<?-@a_l`"
        ")naUj1FvN?e<~(|8rF7VDsfLa7sfu|df)W<UanqL_|U$ZM*44Z-AX=)^2Gdbx#g50<=!_tVkr$m%q4!M(VZ#*_}aZz^>Gh4yVn&!"
        "oaFT$VbvyMy6*3!+w-=qkEzEqaK;ru#fFV0->^SzKpQ|o3pTsex`xw202c>Qyq*sB8B}OM7ey(Zz&RM)+XDXuPl%j%P8X)Gb`|cy"
        "VbX(EZSM~fRBd@7c1qtv8A!%l?iCr+G#r$J0HMkh_J=in=D;qm=czrkV$3z$)ecl2p*T04N*4my1>A0%+-x9w+Rp9BYV-UBGu@ZF"
        "!NTy#TZdSnJBOPvb~nt)M!>-C&!6~HvgpdFP3{2zC7;BoZ>c`{n&k@{t2n?UBy|Gn24U7gsWo{iVokyL5Op=!9Z+Ztlc9X>?Y*EX"
        "3|WBrl=}a!W_UsYV*x20#ScOKBPj`m^SO5#rbMW*pgHf0ZyCr0CES1_V6}r#e$6pt4Dkh&`9wo?M4lm41(hEX*D@-9J<zeQpFr&p"
        "4}e303CO4pp_s@AjKgo@K@=T;J=!SuLcaqPZ(FKDP(~Fu$4g$8m%u!P6PHFP-UzQg9FBGe4}p##6_^LJ;0NV^oD3$-hx-riMwPu`"
        "^En!!QqlB)NV9Q*&>ZPcn?6#|`~j>#fXo3UPP$-e6pnFDA{jRVMt<%fl*>GcBb3e^%JMA;-D8S?9H_5zbx*3)R|6IcX*wW;vIOIy"
        "qrsq&dQi<BvcD<YjMbacl9{sELc=*+pVIbUlX#@+%o=GM!huMIsn<NJgNL|r7WreEAX#MFEg&UJPW$ZXLo0~Za%7#XQ$9aXz;NPl"
        "1tLfQWN$qwx*A#4NUG-Eqdo6}pq}NTjdDur)FkPbO_qMS`!P~FOE!xO24xBe{X0RGAmBJ29VDXyaG(<={YER|ME=Cs2nRu(0qH=d"
        "1s$TZ`DOa@$5-~|B4fA1udh@x2+KxM!52Xy=;42R`RNVDLHPLWCOUlG3;%`c3by~b)98x6ZP=piW?cm)%?KJ4rS);}6#k%R*lyHh"
        "oSBmE)Dz<*ncj!1Bg_~#i6=O~6J8iX8)H?c{aKDf6Z5X#a5_yO^>0~13bIBxPDGcuFWnDK8UdqEQJ;`Nfw0&y*wTWR^JFS`dKI#S"
        "$Lx6w(_LT|W<o$c>LXtYY1b4$AkfB6Z%AJ99gxEU5n4lm$Y+MGl2>ZG-9o&*MWOFrsf&vX2)X_%8C3US_o#yZoR^MEUsHw(62KNx"
        "9cT#jx|A`4C6{3^J+PKa#6giJx-Ky~1;bje#QpdYN|z^S?U8=@>g(%J<cJT%qA|2kd~{on4mfDjX!WCk=2OXu2lvs569S>9aTMMY"
        "$X!Ot3Z6Et3l<-QQ%~PF-TAe^&KS(_q~~yH3zbJc?2@>Z-ja+hBp$c4nf63y4-^oRo$_&56y52M><0t%mmz%*M2Pb23#O1Ev3r`e"
        "x--CFh_wQ1jM)nD7XTB}lZ)P8r_%?*<W3;x9kslqEo;jVLm-#(<~6KN{f7-imJ(~PU-v!P%5J*D9$lVJv-~9<c!SL{a4tyB-M~yC"
        "STVGA6YyJWV3uWQ=w5<?WVE?<zeZHF-8XuDwJl{azge`_^VFyuewc$OCeyihvRp&#rzZ`D@d#{C@aAfX>XGK&`q=i+V(2({na(^I"
        "5^K$E(D#&~`T!PYJt<urFk82x(=ti_6W5Xv7sDmD0EZ0#72<?{*bDhVr%oVabU=?9)s#wjFpp7veK!Oz?p`V%p8fJb%=Y-bvJ8J#"
        "tr)@RYaviYpIO;vUC7eDA1wj8BHX{LlK%e5dzLyK7T%s%I5XHH^nEVjn?;iUq@_yz@B{?E_dOl*h&!R^QMc?fhe{ef(qcP@OQ2?V"
        "i|+8F;BO_deruE(P4SVKG;Yj(2LOT%R*V<OqZcu`vXA0nk?8(S4w9%GdM0Ss4NxC!O?~4-(nmM{nu_p$)PUcs&DUb|gUt?o>}AL<"
        "pPa)x;<BCf=HIQn;HGh=Pn${*t|O>Opwp3WM7bqlF^IUXg-&O+@tOQRU@{c_Z6FH09P+mTGMe?kH+@Qf-<wo)i^&UdI<Eu%v1NQV"
        "pFv@Qf3>;=Cm$|{zhBMalAwCCDtLJ0T0a&aWEN8qq(lsSvvAA&VQINyc&vHf)a;ccS~h)dA%Daw%>IJ0comtwo=*O{aZ1d1_YV;P"
        "q3}OYO9KQH000080000X0MBF+v$Y2R0M!)$03`qb0CRF<FJftCZ)0I>XJu_|VJ~E1bYU-aZ);_4X?kIFX>V>WaCxOzOLN?~5x%QR"
        "{{ahMI2($RyqnEt>~YmT?3532wro`{vWkTvK@BVtU;xmP;^@Dp8xIoXjHF6M7l}l7qaR;?-4vzt_p=|?<SW;r5xgd?s#r^`y5Um%"
        "n`^?1;nw6tHI1kQ9KF9FmUn&2EYC^LY_&U4vb`sk>1&WID5b|o$48>;m9}KZ%&ryN@%f8U^7yFHsw4gl92S}zHRy_)@QyU1<tB*1"
        "6=)$I{Q-|?-<?^_q-m7yxE^(X8LYa9?vIjN(O_WLyz5v|DcOkYsPNm4S2w*9(thMkPTt=2kc*B>`>E&EBkFK*WydQ&@}3F(_KrbX"
        "ASv<odiEY&IX<d+L(pMnOD-!_b6ui2erGf8<<@g^D{}IN7^~OEM+AN}m!Jc6+7S9#ewMWO0-k675Pur_;>8Q{PN1Q0n8yMW`0!c!"
        "2X~pY4>ZwG+9jL_NeqO;>*z{hB<syi*}@ehCpY|l)3R<`GpDg8;{_{kZxy|?gO<UU_r?qy?vF+VK0l!I-(oZmbjnVvSY1XqmLm|e"
        "YAn-wS+Y^6tQd%`9Y|IoV}Qrs@Ga3wS%UF1Aq~;&_Vik*>z0?LE9=RRCq+@b@||!`w)YnBxaE)??tBvRx<DV%=12Zj7@;H)Afg6Y"
        "gA2qt$m<hGqiY(6eaxDKdxdPc#=1E;)k5>iDt%8R>v-Xfe&F_2=^N5;=qb33FtDq6rz}tU5ivs;_#`=aO1Na(me<7?tN){8QZ(FH"
        "fC+Y7fdQBlC0Ngl&S333ovs|Bu%;QE5*rN`XbxmFhbnp4B8<V^4IONA@-v+yfkQV{*5NgpxST3kbP!Ejiu%e+i)K`~q&g&-UXkxM"
        "BvFb7>nR@1gyH0$tR47Ut(DH`NC8AA#llkbKsV3=2zLvxU_&j?!7;)Z!BnKp5@F<&Pyi5x-v@#L2Z0_dp3zg96IzzhB6b=UJO|s?"
        "SRHU-ir3YIUKQX?XR=ii9To$em>fEFW+nm1n~MzOpuI}8G&3-z2#Mi?vhNVIK2r_-J9H%C4Jf`9k}t_0c*Idij3QhZn7uGN_VUj^"
        "W$tc=3fDQ=YE|s`osTnHVFmr)2fa!P*Cdt0D@Z3yhf`+h#^mK@&iMtlSP4}Il7?{V1omSFb3f-?rc8<RYjdFoD2&yEv8C_25_@g)"
        "j<tp#Vz<$Od@SO`V(}16A(@wf{Ib{;hp0aHhX!n|7h@=2#GDdcIQu@2t$#O;7AJ4kfHP!*AJ6{!QzBIJCxqPCIXuWFUXADE)4Pip"
        "lX*p2(Fquinty?$5Ct^VwCWb@8w>aVm=ObMmll@>J;;jLIk(RCkN3YDPj`;IJAi-y!Xvi=S_0cw!i%qfSt0MqSE%WBXc=?}Typ;M"
        "=Rk`WZZ$b+H0R`+ORhl&_yDTxM#1~Db5GKY)I-;ySMVM?!tXR>2QzBPu%YG7$m;_cC$Bf;{D<?izq};a@`>C$>66^^To%*X0YG6c"
        "xeNe*oEL?+F!EOLERI_rL+LiP=*^+C0+tNX&uC-c6D)M7P0ZoH252g%5g3-R7uvW9`rUA1Qy+L9|0t6Y_1MVAPAEpj1^9#Auswm5"
        "r`M`DkHV>_4X9*`w~0?5Aeu3s&XFLL<R`qz59vP{)2MhjrhP~4faDS01@$aMm=jjlCEA7cJOInF*=kPMiN(t^H&dg%tukH@&Yt6E"
        "`b(IXTV~QxQHwfG*B-Vy^6Ni773sw2fs}h!Ci>Cyr~7JPWRU*DqZSR&=8oa^e+Pg9bi;y;yCNCfc*1TE9>=N|duwIrP7a*C{=Y8g"
        "{_eu<so}b4z2n9(7-nGb@*CBv<#bKxV7SJ4A69I5<*$7R%MD}(jN=^lh%Q*)xA$cQx0dmXDw$RBk_jY!6!~&bG*<ds!u<04V2RXl"
        "gG<@|^g<GCIqa1Gt0Ux3X|%8!2xA$<IScs^vpiholeI-K)4}1WmQn}f%Vz-w{)=$}uQ6>6(q-mL@HE9G-?yYzJlPpF&Y9h8gm{A^"
        "(Ea1AXJd{-&w7c2E$c#wJ^O-Oz)ckcQWfrlJh}w6g7$KJ%WvJf#u^J9fsR$J@|^+*FUa3UH={{oRS%lh2~RuBD~QuY>E<3-c5f5A"
        "sMn)m_pJltH4MXMXxpcP9ZR6YY<WIzC|<OBxhAkBWK&L8j<lxdC(HQjA8$h(l~%CE`kwa6r(r^k;fy<_N39QO^kd-#Szpb7mMFn^"
        "ot`dF&Yi<CDbOS^4{Qh<exXoCUt=Nu^M*c)N`PRmB)oSUKm!{JY{#LP6HrnN_ynYj!4yUq0nxcfM!?hU;T$movd4_?;UdggGH_Pp"
        "&7n03@`Zvi(@|=%nLfW7ZHF4`$?q!8YhJ&Cy5Rq2T2To*?wI}y5VmUf52JeYHmvi9m)(R1$3Y{J_bLt^&xmOZHctmn%y&wGBSi9)"
        "Wi6|jWt(sF0ZgOe@Oi%8TQ>5ALd@Q%u-v$o3JXpJjV0?kF{@N-9zFC)=JjI&VT6{wf~wrR3DU;zFaKXB_;AE|v7`I`rEs9@J#ami"
        "ftkVxvZZvW(M&jL_q|Ac^=9HzMfb$_Q4c!%FAu5hg|YuKQG1xC;C}&7O9KQH000080000X0Hp>uo9O@m0NVio04D$d0CRF<FJftC"
        "Z)0I>XJu_|VJ~HNVQh6_bZKvHFJE72ZfSI1UoLQYjgQMp12Gha_XG0|hpZH&FCZ?g9;MKZGIb${giLy-26N*~Dt&wtM+?>wnpMbu"
        "$@dE(TIz|(Yl1-sCSZzzlkr>V1H;_(K`VPt&Z$f2vhU3)eh1o<-X#+}m;ro9U|h81$S2y&gb;OAS3L&@-B_?K1vC0kcBe}0>ne5F"
        "P)X(w;jFG8AAEkqcCo_dNv<Alak;)2-&?%H7MmNC^XoM(7Y+U>k&REe%0<Oy2wo=DS^6dBP0H{|9@<A@Q-&No=VgqP%33Mm3SP#P"
        "_(Lkb`@{t6L7S#MlK<hIb{!KJ*K1vU0#Hi>1QY-O00;m803iTe#NZua2LJ%{7XSb#0001Ua$_%IX=ZO@VQgn*ZEayMWp-g~bzyXA"
        "Z*DJTc42H^V{c?-E^v9(So?3|HWL0lK>mk;iXwL5$7bOcJ?C815AD`~U$DK~0}h2iOSDZy5?zvV;^y+d-;k2^bau0g9@<L~#1@A`"
        "&NnmPN6NCSB|UJVKt_jy&!`6jw&gDI-_sx<AYCq~1mg2>Cz(e<(gP7hQU3&zN5aJkf^Mmxp-W}!`s(^>y=M{v?nOQYkK^Q!E1-|m"
        "jTD7nqJV_<k@yXm4i<|B=+6s3k`3SjJsr4Yii?x}NSFXc<h^#gh&DcxdtB|jBSFFCp`wopc<7(*%s0V(A8lM7;4Ycup+rZ&M+}Vx"
        "6;T+naCd#RA(HwmL@y6a?LqA^EE3>u2`)!f1NKDVDDDX#BPijn0VMQ5@p~G!73V%|B5y|(Zp^YQ9rRZ4z|41jA|-lefk8EWl$fZ~"
        "0o|QP=Lf}jNc_~pCdPiJv2S5OTbA0||616~@zkEJomqcsuk@VjtDkcVn13)xf?FA!y4P2LKbB5&lBEQ}JDBm1LO=<E6ouqSm`^sC"
        "Gt7g|*pUjh)%9RDMv`o|l6bXK%rW(y5RvPGR3d)3*9oSWROf)Zx$UMg{(I|lqJD#(SvT>pIUHVN=d&YqakUNZfA1WB__VcrpiXS*"
        "Geuu(dye<h_O9Sk>aFMyKjYc}=0qZIhd5<c`uDv(o%TD<bUv~t&al56;i$D&>L9!AHeszD=R2RaJbG-Ks%U%Mm2I8Dc8tGEgy(cH"
        "DXU-Yfjt}k;!N!R$X*WSC1(#04=oJWM2oc*v~kXGIbW@YxVHVVKXI1!VxFI|!|sSS#+C%+4<0t9%XA5~U026%D&ZTLWBX<0lIu?v"
        "x_;zQ?+)uL9X~f8gi)}eBDU0cv}f)9_0`gTXD=}f+Zps%_GCP>S6%R!t2A_tQgv%6pOOkUeAVnU?YD#Gcx`n{rnT7Yl4+w-U%gJJ"
        "^`?=_V6T#<*j>F*UcK}B)jKb({{7q6ul^jD(Kj$8ArBdn2QuA`Ay`Pz$YY4OA-R}LBBX9C#F)*H_J)%&a#)U9%n~bv^)lUfbPLXw"
        "i-0J{B)W!W5LJ7u4$Yr5CnWSEsER}g7G~H2wgoc@P$mooQD0KnT2Ha7*6sG$%2qoF7AhwrbVq5DO7KL497H$rmCP4Oyobg`meTc|"
        "S2$Ay_-7^N-zEQ<kD{bBTt}R^(VX9dcLcYJ1|~ec_l#Z_dY?HdG!UnndZt{mkrd4WsswW-Zdo3Dasq-#wEL|yv972ptlwJ;gbQIu"
        "8P$T6^$w2jDXQe2gqq1^B#a&#L|Tb6+ym-I`W)`*DQ>!ysvCxgV~r~pV=Ij?{_wy&tfn6P<Hz{B$xqGdS@2yvR!=CKY^a9J8c!x4"
        "R_kSdJ^p9|a?R3SL2Wi)!_@AtJ}m7~-IPQ=-nb&H)rZAmGPXy_EGxg@)&h7kYdm-1Da-M*RWIxdAk)2r;EKE4$I}U7?y)Tn+#z;Y"
        "ho-1<rxmFm;CVx55Q&QmrcSMr>q^Myoju)5^?q);|9t!7GyDghZqic^)U?|Dd|TIpZ{k6H4$42j{a!zx3<q%h0czjYOUpL%%f=(R"
        "GIa0_L#{}>aDhczsD**7)w>xL#U8Wo#+UVEdgZ+LVz+Dmt&+so&!JcqPZuU<l3zcEddaL@hV|#1_cZWJkX#B3x00iQ%c14Cq=iSj"
        "+D#hfP5n}cmGCMKbf{)-9Of6zEys_3&Ir7+3gh#82PWugh@J-8lP;;0qp)!*!eaWT$i4AN{Qro8&BdW|aUuDui(!uBqTo{_i5?@n"
        "kW0nfjE7U|)STHZ9V}ypubdwZwc^s;KWfg!_2k`!v!hnO-0Dc|F6shN?!I9th@^o;c-WGL>gnH~jG=~y$o_Z<Z5Ym{i;2A+&u1_i"
        "_utRvtMzzT|C_OOp(iW6jYsOH$u4#gCLGB)$J^Q9M=Dx{!J*EyZaTyH_SDe$rSARf$-G$8U&iIt;!o#aGB4EsUqI-^P`{Sdt3iLQ"
        "rve^dPVia%<r)3bEwI#PTdP=~7fBFZRWFAxs{?)1ofqKYwxAA4yw2w2n0)0WqAV7_{VDXXZ#7%?sNY*4;LRgS770~bJ@96F^E@-F"
        "io8=w_NMnGMA&oMhnrrf#v`e_uyRRi!AE^ziKi56tMSsyh4_qJ6oOh=2AW0&00DIoPqL%+g!lV}+e3W+&n^*ogSabPKv0J?J<4c<"
        "3009Yn27T|LFY=A1tkjaL;estiR&I}rmB^bh}fIw6-1IMLjfuJ$+46m4AC;c*+OuE#h3ukBV}~jyp3azO*1vl5!RwKI$SIIt*3?k"
        "KNh&S32)zYN_~{OgS?x6Sc2x~CUzg9p%rkvZjS8ui`K*y^k*Yj|2oG3NU6Lcfm5A*%vC0U@rlB`#`F>i6!A;j%KKGQZUGVZvLt|C"
        "Ep$an`}0&UL(>#x%H|h)EmuM3yK}EXE{0VzM-6y-u76e^KPw~2dasv|i{$@;mrqN`=RLh7Auq^CjQM{7P)h>@6aWAK2mk;8Apkd|"
        "_s$v=007rV001Na0047xV=rQ9W^ZF*Y-eR{ZDB8Ec42IFVRUJ4ZZBtZZ8&Q#aCx;nYjfK;lHXI6|G?vYI4YqP$!}|-*_z5ZTe+P)"
        "Zk*Y=Do4elNJ!$DA~k&Y5tsk_bvFQ_K~lEj+<I#_BGG6x`rQq1UDy4RWSNMazh8Vj{fDzD#3n4lD#<e^%wneysR&E4S5|kaaLOnz"
        "l5FWarGygBqR7_{&PfXA;{Dg5>$)#rynK<YH+fMxVX*|bQZz4CVYy0^dGq|gWu7$;1;7!$C{I%nAq}OY#@^+1R*6CZY{ClB=)o`W"
        "h^zi_{?88=A3lF_rjC0s9*v`u_-J&pe>grB`=|4>(daZhK0Y{}$KrH99F2}o&-NoA_u})Pf4n;n-hcT0`~s%lzkK1qpIa4SlDQMd"
        "bH^u(gTs?JIvO9IoJ2>X_)sk7rw3<8XYqUvFOK)mPQrP3dU`C5jt&on*bhSyAB^@7Psay`$49R3oE)AU41A4QlocU9I5`-_2jl(4"
        "u{bz99335<g^@TsJv|tUqw(nYIE>&wCkH1m9?nH{b~cZq`I$JJA4CUd@%RuBj}OLYi1^{<iw~bJzW(v&yRVW1o?7f8Pp^ejt%S1%"
        "Hi3Zic`DXUS&5A^zoUPL&N+zsCV}C9hd*Zl{LgPdCtw^WE|M%vovdDqB8fmP${SG(fpG(5`pXxwSUBPq*cDZQ(w9I2c~QRJqyi;5"
        "m?(N&_hVQ^D?+TG|D%K@in5Zc%c-P&Fnt{Y)H=To(_fq{Z-@}^LMj^~T@%JBKn_b07SYNpT*Bpj+xy$)Xz%RX@S8t-Jpg_&isalv"
        "RCSR_6vJha*Bfs<7?xF$Y`nn`R67Nkc&-m~U3cIl3$>a^OW}OVGhvu;4Wum+5eH>n7m)~x{HFBcqySyci#xEIO`24i5#n|O?-6(v"
        "`sOJ@MTf0$rM#qxGdV8~s}RMDN^g)4oHvd~V9smD9im;O?f~fz<6wRV8t19~!UmKT4q@f}qhk*-4u&gn8z)Q9OmE=)+teaYQB2$l"
        "6WI1Y!?YIXMUfYtyDZnIS9-^YIantpv1sA~EppF_)xZxR{Sv7grg<2bUYcZL018()_{&K$+TW07;Ts?uDaN0(hW=D!9^$eICC=?m"
        "DuF*jt5geI#t3*J05sren0%Rrf8JDkuoVJ#@&)**bzOyEch07sr=Y(7b@BPr@4rZexg&Xq(wPh3gK)s+SKwrgb}KJe7byy@5P#Jm"
        "B_xT|K`#%CZK*^HV7R`5-yS$*kyYjNYf%f-yQHjw{E8mc43q}3!goAdOM?MQh^yu>sl*zD#P}Wqb>J{4giINIeF|!3fmqqZA<<>y"
        "#lPJw%$^hF8KA9&c?+Bb$`<absy5~1&6^^;87{&5)$_U(1;PV|hTtjQ<QtKN$(v;n#t;LeRi3AN=tke%>sa$32Lm}6&>*Vtn|q?="
        "LG9EmPHxtg<q#fmB@h1fohn(2ysoClqk$8aporxL0-<PeLb(AwN`~&82BXmk{yVlAi}pfB?)>U#kdXsY{s~9$n1#jpO@`(z-9cUi"
        "W%6BcVRK2}6re)NMG*nU+}7b0h*&_<0iFVbCdnH(gppbOeg#5b<)F_aW`AF5u%j>zH;^)rI^*Q>7Yu%JO^Jn-=qRlM3(P?_Qner="
        "rJ`g;xRC2W5US7!ypivm4v>kSe2?)88EuFgIB(^g!7g420fTi7o1BZ_yC`yGoJf|dN^?%!h8Vlhg}UFeZy8uoo@AcE?fy4v6YOZ>"
        "pgRK^F5SP>D|d!=_bvO4qN@Uca9PhGR=#~ZeD$|qf29!T%Xu;iyE7&=WF(B??#(7ICq%h&&N{|qo4NwwEb}x*hd=Y-zs9-bi$y6a"
        "Un3I~)FGQRGUgo9JMeX8FvW?(rKmi!yMb?BD$;EmAhDH6R#ssK@!TMVNKax!0R2{nfBl|Kl*r|@9nAq^49s5!8aoQMOJgAx0z5Nl"
        "-gH|wCI7MaWu_%1r6W5Mm1K}XC2`Qq?;$CP!4INk6yz>x;sj+hxtAiGj1J=m=G&y)hv5dCS?t+_+}n?&ubQht5<@<0mh*dHi#!zo"
        "<YMOM_5sXc7prwxTp<+E+6&gG%*1Hc7SamoD*%$E)_uSfGHxlSZ;bS{GEx*$^OBNA*Bx|mytb^jZ(LK`JGU|H_MRWiBU=>>Q;X?-"
        "yBHj8OB17K4|amYP2Fz?6us{uOw4294Xo&59KCvjWt}!Eg}#N5G8_W4GTSX^p>A1<Vgn^><z1r%JdWJ3hZ5AdUT;b{56m4hu;5DE"
        ">AXo5z<E(lRc$cAbX)IJdBc9PWatqxkMvX~!;%;X{=n0|&?ZN8rYE0FS}&jbQYYyI-a|w?r$OZj-a*O&??WdOH>&<uE10Cw?Q!Wo"
        "Zefw!VoCtSSYg^M+MBC&cPp#mf&Yw&RG{;jl3nz_z7>!ZYq}XQOL;NiXW?4-vQEJm@43pO$t2EKT!3KXxvL}wLkY-|^e{dQIOCO!"
        "AotbS2P)w~f6?-gt_tW(`+yS2(X$frnIJ5qB$>Vs!G#Q@$-6uxQ`kgc&_K}psasczy;HYi{_8MH7Qk}T>7ca%6XdSwI_JqEkHT~*"
        "p!Qa^4*J~Sddz)`tA!)3Av%(i)G2Wf6eUvtat|gnfoiVQ21z@Bm%c;K+scavj&oZia7pkm!WDo9VzJKApHsRQ807`42XOPu=V+&h"
        "2rnrm;%;)rSueF>#%DJhMww5vgKM;bG=WgyFVeNPs`V(C%?3zjt31k6T&1N*y9O^8R1&Jvpn`()3X}$bNx#})pbk-;qPyD$Q9<!j"
        "ZX1k+)JY1=g%I&wJ4(500U1|>06OO>#8QAzDr`6gVHDMn5q0NL#wdeS2{}zbOzfdUbMC=5f=idh2oUPjiWwS8S&k8(t4ByMu4U8A"
        "dNmaK4GgN&eQmg1J8gHP<V+JZsGgD$gZaQ$h&oDZC_qt9ly}fkF3+oNT><6=SU1L_DLwn6U`yNU(5>SGWepX!A*7K@d+wTdgIbw3"
        "MlOKBQqvN~Llf7U8I;{CM@6;iW4_8Emu`gFPHU?>6(JB&YY9fiF14GU4cb?UC8)xkJJ>>f+iTM5P2U<ww!z8O$*q#N+r7O7HHMt5"
        "^{TAXifXZL#M{~b;C{Hi$$Yjwm9s2qk~^hZO&sKGzf}HmcBzMFJC-pBZ`e*<yk-T)E7f?kakkr#_9*X3c8Gt>Agt!4jZ<|p&C6L|"
        "u|=dl9&vC(GmFc7y2V)&$7ZNs*rI^S&13Kbk_8U%m%9p}$iIV@Xq#$P-HnD#r;e>9)GrCMyH*a=6!C4%x54s)=Az$_+|Z6js<NtD"
        "Fh>jvKo@MV$BWNpp4PHq&?9cfT$VQN$OyHPjzMUEH$@Jyi<==KAXC)U7o}~M@AqCw2;X7xM~4ACfzy>kCnb*)d1$6d{vnl8NlfFn"
        "m|&;{VsM#(AJHdkR!du-APqOD8%(=+K;ltbBN=~y8I*!@mgyGT28xe0mIoNfh%EEYrivp3esMUZMJ9wHGZ=1c#Ey3fEXggDL1tpW"
        "*3|o&-@#Uq*azd-yA)3P9F6RxwHUv}tLVNbmz*U?93)zHw3j-&%JLg(hQ&2_%qWBm%23QLhR|M`rDl?F2|(vKC@pnF>{$`7Y=t~b"
        "I(*(V%bf4Yre#W)bVik|54SN65pCt1+J->65b@;G%zU=Drl%>75*hCelK##ftX$R94Au=s#ytZnQjSYyAv6{J;pcxSCksEe;$MnP"
        "sM<-O2rcZLlL>Re3UgZvo3VBetlOeBu)W>Z$8^g~Z93}IpKGnz@c!x3>n*JDx-_}Ev#Nqn7Xr4bR#t-Tf~dR-92utVE}Jr)8tpY*"
        "devBj+)T}A5!_{VM<enLY(J!QpQ1ImtYx(d(qx@fZPk!sM!UP@%3)2JIj^VA*vwP#l6CDtWvn~3B$e_`loa3%-7#qxK2fY{H7a?t"
        ";~CW^RJau;;4-eMHV_-ujGr`&(QmB2K!dsm4ie0*$Pv`VK0;J^^2q{F{G&cC{W9gQ&K#s=Yzok6$WkEZEjJ>fUY9CHk-1c<<BwOE"
        "ecT3WAgJ;yk(JZ^(c!6e%tnXV`d$ejb|AI8`=#mNIW#Z%j=>@<1QoIHHb>8^?v@mFmf>v@cFu-=VC=k|_E3Ot#|Pul$^H(}QyZd2"
        "F|T9Lbyk{fO*VCYY1a6eOmp|lWv3hxwu7F>Yy$<UI?sh&da0K`cr>KLiN$&pBKIniiV7mP@D!l#ep61Nk3?p^6UK3XG$v4(D@CHv"
        "+C8KO1eh9i<-Q~3dQuJwxJdIGPOwr?mh$KhnK~K$vy|7Q%6bFo2&k#%p4!S#X=p3tJ&70lwCXUnVT$%l8Q`ZZK^Y!8!1OMNR$&DZ"
        "3!)F6ikh`9@0;c)-c&#}Q@0hUjfsRjaCf66H*L4WP$bN8%c=5cgNxW|4q8Y|QoTrS8%5B&(2Ah;*Q)}uRW-1zUZDd6lfN!UGc-z#"
        "cAl4Vww+~e(*^|J<lv$1THQ4uPak10|0vm^i!0S8gLPB2W%8GbT$wJ)hsHbtyt$(p^vc(bw%Z^RH{64=ksklbMk(Y3YWEuEaCCMI"
        "Fw71AT~RJCW2zfQl0@Z?vs@m<gL|p5jzA70Qb>G6L|)1`U_!$CYD2n_rFJPVOsE@4)RSyZYZ^Xm!;@qxZy6fZ)2L)ySnXBcZNlyu"
        "3Sfw=M@7JYcPX2266srWze{UC6W{_Y0{iKw$T(~usejrM^1%3FauHT0_$b;Y)b^uAjc>Vhe$$YdUZ-@QaLa8J97yX#{#G*kSRT-$"
        "q}^AnSz>-_dqVGKNw>8;H!)7X{YGx4^|EDK_G1rhuRU!W@OyU!?8E)&y$!WHVmzR-S>6YNe=qmOGfEZ4zB3*j`_9p*UCBtC2s;5_"
        "Gj^8<u-U$|9g`V%>-WwPk7-Lxq+Zb>XA6oIW|PL>DtQ5AP=Q?~n|>GGM%Co!SW|WfO`S1ed}QzWsZEc2hJ?`rKUVVVmPN9pE6OQ4"
        "8wU(2y4vJP#{FMdS2=hB@QgQfsj7UVZ>doqL%PQ(5@p$Ph0;OcH36s^Lb`URd={8^N%t|7w~}wGyogr%UDpXhyj~>B#_d72X9Tou"
        "4w*<f1RJ0uZ|n%j>20S&`w#G9-L(kXhD8wNq6*Z-xH1425;RWP1a=jG)NV1OUFl;aNN$MvLv{Cn?sB#ciyGg}0Gmq*bi~0i|J-#2"
        ")A68h0-D<jG6?$on2Q~t5n<{UMm#Xloh+LRL_(V9!8Qr~c!+#<I3kkTapR|>Q7_irgOQfD?Sh@RP{*k&bX3q0jCQlsCU{ms$1+#D"
        "rsKU&(g21u_}4{5sU;cPR)6e}MUJEbO{b4#zxFU`*K4<e*SVm^5ky^(xx81dvLaLmJc`%nz~m&bcDROgU*104;z}5kB!etMjnh>F"
        "njdDMS7VejVqkV{rO3K;(Q_FfI;(%eqxB{l+g9yJjQsd8s^f52h!QJ%5khJy)_HLkV2o={_qd;vcZ+tuYOlgBB}pUsitdMVlyo!P"
        "z3sa*fcL{!kZ7d5Ams-qeKn2Ej~>l$WnR*iwmPRh?&9&}tVh>-COm0ayJmG=BjWj7+n_LsY7X(+H$t}D6|qrBDad;uKX8F-(2He}"
        "*5#^wc5GC|Zk>{h=!OGZ_TFv4sne)_`*s^sigxfY@Eux`YbHcTj9>O8F_YT8`~30C@8@67|I_M`QfJaykPTvtL_6YKoNVJRpZH`3"
        ">l!~H`ylFpgV?HvP2xut(zUcsgKtws*U~f%y-oeFcSGHsya2@|+v&Nn<g(K{b5~7Wrr<f4H3xlEn4<H>Pd4a!EX-oQ40o)YSOe#E"
        "aiGg9uAfAq!E6lLNk-thP+1y}?1T&aB7-~+d2gR`1IlVa`i`7vsT8F9_(4q1B{!mojsXf!QWpd&B@cp5RRU{dCEgtZ2$-Y*J^OuU"
        ")^bzZd&j^Ui|Yi6EmT5l-|ahQR+`&?LrhxPj`zdNXE90E<jZvSfhnK<)^_-tYVv%c5ib^HR3sbJqiGPtc?8T+DQ`W;6FLGLQirad"
        "%cl}&Ra2L~zszF%CPe&IC!mnrMW_rKaKJ=DAZL578gXqv0?{of&)wUTSC#gfyJ@gab+-{yEHxfovV%#Q!t>|Lo1<zv<uK7I2O*bJ"
        "`JE5ue~YkAtLf-5OiXY0_UP+Bz!{PaPqFlcAW#7@@Ny;6&D4EQ9|&&b2iW}6Kxr4!PE@_B+s|5fc*&uM_4MB5K%LF%%I9fzZJpm-"
        "LU<;MUFG#u{e1pEU(Wyd@bUc9*WlA{AI~pzU8NJB9tP<l%_IH%6KvXwf<-Q)QEi%k5T}YNdn_0J2?DdnkUce9Wzo-^-K39uG6X(<"
        "7rScHlWdUTo4WyR_H^>Oqhz*Y!QTB-SRlIYK}xcx5|t;2!9>6~ary_b@vprpS5TguPWH8Tw$BO)jC@urJMR1FnSk4$WUz#l^<C_*"
        "U3w?_yx!3ty+}y<e3BQ0_Elc|W8u^uz`vfa&d5DC#w|19-s_U>Y@nH}zcVl-ZiD>MTvz`+*smA|bgL)rlWgPLm<YR#Q;oqo#@y1c"
        ">uo*k(wUR-$!f~cGS)?_9oloP3+~LC@z1{0_w4(<dPR-vUva&6K{pDTD+j&hX6I5guSEMV_U(E-)l3QWha;WCF`BRIzTIw`wkQ5f"
        "9j!6+6h1The-bdIPd^Di(THhJ|9?o=jgGd9^DC%7<TVl>e&Dqd!fSmIL6=eVwVsE8O8;Rbpu!^v$l!wji-91}1}{s4moNSgP)h>@"
        "6aWAK2mk;8ApraYW(lqV008;|001Tc0047xV=rQ9W^ZF*Y-eR{ZDB8Ec42IFVRUJ4ZZBwIa&BdFb1rasZBk2X8!-^R8~h)JJvHEN"
        "((4`yTVj(yNJ5;pM>CdYZDBpkjFdipeMh@>oYLqZ7)kTJMorT!uRe8vCKvR;bVSFnW3(>rz{N~FprIDlL|~JV%jLKJ?rZ;i`*7a|"
        "+cr%zpUr0jrvOLHd+)a3!ZC4%M=f9Pwr)rU{f<Jw)})wO8RhhJ`GUS+b}2qlJ$SE43=Tu`_SCZ?%!TPb<tIY>Yss9r(#>baqX?|O"
        "FTYSU`y1p)BHerjx-1Pq6B~V)j_MNf!vc;>)|o0pHq}fwdRTzPj6!+C^x{NsJCMu^;1V~^j>Uv@0UtmA=cPNdXjrRKMZ<M62jYql"
        "i91;Jw|AT8hs|TZUN4~TrwTBt<2W!XOEF_JRjblgt0&~^N#UgeLcr(-5=}Q0$vRS#*j9&RGFi}v2N_|vHJ*Tc&{SdyE@DpHo<VzU"
        "Vu#-SR;h_Kp86jwm=uo4PEu5`ttW5-F=arIVvJ@z6n_d?V`%>g;9M{|p?AmlW-w!y{?KLt&EL@W(?EfKk|~v7G{YptXVt->D#qzT"
        ")2hmTIGt?!hWlbZ`vXu*0|XQR000O8001EX=IZSZqzeE5a3uf$EC2uib8=%ZVrgb?V_|G(Wo>O?FJ*RNY;|FDX>V>XZeet3c4c31"
        "d30!RZZ2?nwHohp+qU(4?(#pN?1z%8%s5-S+o@GL*OqJDV##Nz-EwY@2a}M+GexohXj}F2zwbQ&BtZ&Pl5SHj5(zxK_wat>!Jg;M"
        "sZ1Xjx!TE1k&|1N$D1wX_e8`6PxHGGVNYe2#;GK6kuZ2;aV66tACYzX#1fLQyx68Wl?4~!%NH+SoDfmnmb{3W5U(Cs{y<Ve@<I|y"
        "1gzz$+z~<Z<hFPU$%+Xn2(NNd<k^lqZdgt@`>ldumH;}>^AKmcEej49glwJ*(L8e2Jhr5pA<4GoI?Y;`s!EfWFV?)+l7vbcXH*C#"
        "NJAq@V>xQ43}{K^CQEN?uq${)&eYs(x-Mdx-LdVKhH;TgPGi|%yk=z~QjpJXnbT5i3h7#g^we7RB<qzi&5Jw*kVu%7(T0l6;Bw$G"
        "-dI~H%WM~^!AKSj4pgb)Fskx&ow6iK()D@-N?Egr1GBh9ZSLiZzc*s=f#&ZlpR9Prf|oDU7}3VFqT>FVi7J!hmoEr>(p*TI$1F;d"
        "F%gpMQ5chzum&_|$R#i_CbvbA>4|lf$7o})@e;`BDe`YJFLI_KIHRJ-6)rL%Xe63KQdH70$~d=2@n6E$%Enn#D%00;^4%L48B9ik"
        "V~0)+NXZ%6PimMmqY~^n5;0FpX%GLYUZgqDzC~{{W&?r$u^7CewE?lAfByEbHVn<O;xR%8jT#3vQk#%-nh91jl5QD@C<0F`@&qKi"
        "&I&34=r6~8a)zk~#znmwqXGcOA6O8Oqc?*_u<lbM-%rS~nN>=psbJ(6npJGddBJ_JUa+k|Ubmnc6fz+9f(GY+Px$p)K)#p^%j^(w"
        "_5d#wMb||gZBwyD<EXiid}krjVQ@0!ZT?}m^IL-nf?5MHD<$z;-bQ4xG!Jcn3s+iL)2A{8905cw!Vt*bB$E}-_cY{N_B|rr)yeAg"
        "B04{ry`7$UBV#Z152)Q&XoH*ax5vltTTQ5#?lQcO({CYq-Ui$(S0}64X|$NXeFw~7xEX14BXzE7D3K%;=q^d<wHUh$TAmleR?^ZB"
        "K3hJs`T6xpG`+rFT<>vI)R_|r5ksgplrPCoMV9C|HzJ!ffn=!w1%U)eVo)2)WeS2OIRi%{@CM&<!^jpynt^1>N`@NQV=6aIG7DD>"
        "V?5=%Gf;kvfe0-LY;oe5yl{l}eB>#+$my2Z+b#<9dCXxxP6Z)nqR1YYA6Rx>aDt?O32oyczKL;HcbF;wm9J2~>g5n$v5HTDlZ2%w"
        "1PW6DYcr(`a3O^==+XPA>ms6viI|p6)UbI4Za7Fe+}<ZC_w|dIXq~dB6pW>~uXEhq+D9l}SaiDs73!_2d=1#3Y65-vUM1H@fAa!s"
        "tw06m4CAo(<LrEKdh#}!{sLd;v-xa!F+Gbs^0g{SLj3y155f(5veqk$g4IUAz84w&MrP9mccOqHFD!Sq48A1)V60T;wT6l<BUtEx"
        "y;V7+(|CiPMCckVkmiI@4h0#AM4|y`bHq)CHa>`C674zM(lqxGc%s5afHmJEhmZ)NQ50lN(+rBqS75YQ{Z$}F0g)xmGBsu4?8a6-"
        "!Z71N?T5M(%(q<H+MJ1+%Uv$%lQIieKs@C1`-b+(VPKYnF4BZFDj{{UN7b|;v`fNztpypOI||tDvsVv!A6hUJa|0rn?E7nuB|4}J"
        "h4o0Ydmn|@cOme&RMcDZamh$6n5nAK<#1i@!m5O-kU8X~utSInsX#~q9I03JvNmtX9}*4qK9HLRW&&Q)?0X;oe&{%`k#ne#0~Kc#"
        "HL&0+>rrrQwUzMyd7DVcy~TD=7ui1>b}!M#aDQ_^yW%t#Kx@l*Ut?I=K7e3kbPDO&7c7I~9$9gz7H7jdmNTBlky@n$891ClM94D+"
        "`vH-3brE6}uN874A@kl=(uEd3jgoLlNF^5P!y>9zxafuGfW{2wnAtvspjs1x`@-(JrtZ)WYTF;)sv_FMuQ9il<r40?majuc_)*yI"
        "+3vC62|EAHy|T3X#>|>IKy@Edeus6ihauT}4>9P*jaCT**fy(uSU1s&Mns8ag0rWE+O70;a32urd-t#jw}>Y$%3YK}^1@6#U_~`Y"
        "%65lA@Popg)I#^uWYXqFI0nb}kxDeMlUP0}xVHsO{2&+|Mhh|!lN%K3`;S1@GN0TWf;>b|1ugfqL&`>!xX2C@vWW$WNt9KtQzbq_"
        "?S>~9u-Lms_QLN3l_1ofud;=$-VNnB95wcR-mCj9JMnAZN7o*VhPt531>@2`9+78Ef~uhVf`171)_54ST-t-Gwz%VT8;tkYXpvy<"
        "1_ljWs0ZA@;2H)V$`f^x1sBpIp(@L-s~jWGF;Tfx)^m?kEUOp~UF$0IhgebUsM2(~6A<gC;70yy$6hT}8m{+Ypuj0gtb6XV;ugHl"
        "-^a7|@F3*J#kPc6)PJ_j0AxHOUwyAn97j!4??`R(%~vpoOTP+6MALC0FL??c|6CWadup?bTz%}(<uNv}d}BCJQ)6WerZ=f;W4nC3"
        "O!IULbtuapQeNa+AoU%{om`y1`YDwc)$Oa3t6ArmgvmAEFeH?at9Pr5#T-tr7ghnOPNd2ap7CIN;o*_;-EuX(yk0C;`1a4){A`9L"
        "oOimoTwYC2J$(kgnEp%6zdeiI&VIZ;xqcTZ1Xlfp-Aq6Xem~aXQ3qkRGtm;GaUnvppAa)XXikM>wu6GAWE@}fn#(R~cg1nBxSUQJ"
        "@971LpO@3?tLw!%9Nh<DR+m@9bJIhUmscRD!D%i2`2u-Z&Q{Zj_u6xUT%0T~mec9k#5;Ba%ofx6>EdiQ|EV^Uk<-pVjzjIKF4`Ij"
        "&rK{8aF&TTKKFi2zWz0YFaL+J`VD^YMnf3)Yt06!2_CMK=_em86wc2+Uep|zo`YdE{CKInf2_ifS&wIlwynggq11e!hS=+19>{s#"
        "(X4L<-Rbz?w8=bWMkNNj?{u81U&F2zA=?ij8%o*Wh!glfFdNenRLr8Hl4T`z<G>Axaq!^0Zr@E_A3FeozCG)f;2&%Q)}Q~j-8Ik)"
        "tUolC6~#_RZI`+qU-#Yi9I6J>Y^m!dlXo2kSR-$nyWt%#D!_}2^ej-KTng-z`Fl{<59xP=oZY2A?89z|t~}i^2h7BD+Fes?&%^<6"
        "S+pinH>$^Nq<HSj0SW^n%4hxiA#`zN32HxfG)31(!^X{iV9@@rY{@inT%H&|CKF6Lj;Ub>{WRx1<e2?(#RY;6ASNDX5OZSY0;yfQ"
        "cmL6Y+;3hDdi5uEu)2P{nq5v8Kd%}eHVyr{i>?~hhQp3y7-meY`P?qr{~(s&ouA3ws{FR%gDTGPblvn-)4T@L(Z;Gu)qNv2UawA;"
        "OAjNC!OTDq#KQ{@D|5oMcd430^Fp2%Ri3E*dtC~^GR7NE_m~DCdZKC>vBlA$th&2jxLfjl!SCOi)pr4r)dxq6aFC+;9ZPKOe*Tq="
        "Ycu^yao?M8s4sF~ow;DJmPk(|8cH`Adg#2qpLLWT8fWxtvjEfkehsG!v0=#@FSt4S=J@z{+z&Jzr8!^%>#Dfg!U-z<okh96pU`WC"
        "Z+^!lONG4Yi}k&_A;k;6_Et1wx&N%1ftHjzeM5uugSyK^a5E4byuVB;{6h^mK-gwrkr1^RP>B|!%VQh6o^8Ajwrr-t-i16?z)HXC"
        "=KO52ZSbhe^^fDhbg#_f5H4f1&+YcTW)t4H-H@d_&!VMS0YLp=L+RbZlztdMfRhmi(KdkOl_f?{YWt$1liB;k8dtX6%jO>B1;O}("
        "HmN^I#ff-bWmyvsx<Vpv>KMAuWmh!ypJ(i9u9Pan&fjflz1b?WneGHR$EcHc9UYheg3qqtQ}fYA5%aXiO0TDZ_c9t1U<AZ(miW&N"
        "p8o0&*76TFe`X%oyU~9wWXwBo+i)FG;IRK6P)h>@6aWAK2mk;8Apo`$?x@!R000&Q001fg0047xV=rQ9W^ZF*Y-eR{ZDB8Ec42IF"
        "VRUJ4ZZB|hWn^h%bZKvHb1raseNxSC+b|5i8}K_2I@wra3_A@TveQyv!_XD$upPR>kR>{0D@z6?F9o{3`z$+coGdL*BKiGL5`_@!"
        "x9>`DO+$*Lfy-a(WsO$dz?aK!=U-t<jL<l4W%vLV{~!ZdWqOUUaYlpWK#f$v7ea_MNfT;2#{q7=vuV<>Ye6|<Pz7?<7qV8cLwpX%"
        "vNpIwbu!!-s-L<SLmK7AUQLZt(rmD8Wucr6937va-?5X#7mS)lmq<$!jSXy>0#<&D<Bh8B8RE0qxqKew>R9S}G0s(j4-9|Dp>kM$"
        "gmY)HOcNM_-PXejuCAxlBibSL7_xk-437tUx8e0*?mb~*P%~>3WPOlUVMRJpqU;Pi8cXsBKc(q$&Fna9#HYS9q+~#t(MxzZt@|kQ"
        "y<#*dq}JJK><k$%T*$6NtFyz2;-43~Z##d~i1F}$DTbJti1LLOSiHb(okmkx*<$6mia&KIvG2dGFfi2-bu67=J)Wqc^*v)Hy&_tj"
        "N#l?0X0Nw92FV)?!=l8M%Hre)jlrJLhN8XG#MzYi)er7*0eDY7RPJs}4_VW`2fPoN_-#dT>&RyDzjAm3V<~RgxUo1)2}0z9o>fkh"
        "-K<0(n&tbr6#oEFO9KQH000080000X02JkHTtEN-07C!(044wc0CRF<FJftCZ)0I>XJu_|VJ~TJW@U0^ZewLHUtei%X>?y-E^v8M"
        "Qd05^(M>E*EJ{@<D9TSMO)kmI&r`_EOG_<E%}Y*IC`!#u%*-oRNCXL%q^1`of)ym?<R@qADJd!OasdEPO9KQH000080000X04kA3"
        "%HRh80OT7004M+e0CRF<FJftCZ)0I>XJu_|VJ~TJW@U0^ZewLHXJu|>a$$67Z*DGddF@%<kJ~m9e_vq#2Ubr~st9(`7FXvK+)G*%"
        "2Ly+-*OzS|&=O@6E0I2u+U)k)|GhK(kSJMpHpN|m1I~F^n;-K%znNi@B>8@`CHg3pAqT}+^F$PDnPQ@5<iG?|)Nm<Y5h00T{Yfg?"
        "k-z^$YFZx|k&3kRo)KM3#dMJ*$?eVUO{-*28fs|WQLULK9PgQF2@m--BUODIFYS3NYuX)H-_xR&qUDEh`OV{rDc&<-emJrE!g-@;"
        "ZAO0Y*okuW=8^WNj^+O6KNM?tjXJH-KoJWg@0sAgSj7XaR;r$w3d|xKrYhvc9!*X<Nt@~cBx^ZqDu0uyHQKtSZ7aJbko=ZWGbr{}"
        "ns<ygPj8fx3N3Ma^S9AbDUkVviSkD^u<Z86u95GopS~NNv%9?^@UhG*-sl}M189851qfIa#e27}jXYI~Q4KfXhjObyBYPXJ!X>xS"
        "Ft>4nc;Juoef-z3)<;a`j;8Kj4_~`{PJB%$f5el&a^i*m+tbHNun(~$fsNFw1V~BJZBDkEuXFO%CX478X=qmk6(fpp<;%@CU&GOi"
        "ynKDh)DY-ztZGDJo>0+1*3_~;4TeE#3>uQe6DJV1U~P7KMfAW8-FFOwi9p7{6(blp^u#-9eEJ}H-V$%jbG^x5$#&WULPDCzh_kJZ"
        "iFDe^$mdI1uP++I(aClbJw>@@<R{t<%!X>3Oz=?chvvW_JsxYY7i*Ht-FtxO-wz#CPheLY7=>r!v%9b-Dh)X^s@`*vrZ1w<$;pdI"
        "iwrEVLG3{F>*rpVkpNdU8S@rzR*kH{s8!2NWo51=DnT(6)2idy6RycyDO@PTQGbo=o@&-{!K#`HWC-!V_yeovDH&Kc?14k9l|%sW"
        "q>lTED%vx{luom(a3@9q-Os{2o#0kU5;#bc=Q>oee+?Wky#mw&bfX<ur>Q<LemI&@B(U}amxH!u8a)gk%mo*R2o4JNK)cl6l7oRf"
        "f#Y``m>ed-LIMGMFvqInJvTlV%*=HkxH-C-QE=hQl;xhrq3WRrD7vEt%n173MUHu&lLjqQx)YrghumANbbqoldSDBEKN1CRYmoGw"
        "rLZ-pwwJP7T-OHds7D()NkSQM6at5AHM%g(ZG^E7yoTTA3@%0F!Ym6_&gp1yye<AE>9{sAe#{Rdh*bcY5J%#`(mAp$y<sJ6^qEHj"
        "w@-G54Xt%|SS0v}JMiO>2nYajQDc<LshlKkqryC^I(gs_@c9hA{OSr#w*fSLNwl({oVkyH$jx`VoV?t=Uoli`y^Uh0KGt;$DP>sd"
        "85gvJbXqeqe$EqHAlM>-DoSy$+KLv|hoA)1L(j3G>VSCra+?)ErU&Nd*IZ&3^mJmPNiC6;Q=pxr51|tZ_XRyhTrgmYYA*-TAO=q)"
        "0M@?RxQs}ER<cfK4}|W|w?gc#2@KW(6UWFCVOu)Gy)1wjG;`vtVMuLDk~-65xz{PTr8$D5EMFGuZ0fR<ZCs%dg9gLkrf&{3G{O*1"
        "p}(Dqm~GCX8s@|UjP3WxuMR+5*5|hL&r3~-tq*iQQ<w9WL2rvfL+%=g0jTjOYU<-F@E(GAPoJRQQBdD0*b~c<VIBM!8i*2Xij3lj"
        "{Xvka$MJnY4j58i>Y`KBuozD+6dVmNU>BH`7n=0cTgc+7UmNSSjUB&u*0W;&I2C*=jNgp{!>>F;k@4iZBInFq?D#c*Jq`Nigf>lu"
        "grPbI2iiFyq6cCxBh5E`i@txJ9_I)Wj(@!8!;cG9G~b+V^3G+66c{9A5Y|ql;cZLWp=)^u{m_9J7%jPgkR9xxMg|>LLvah?^s`9L"
        "%3$Q;P&5~%+~UO*QOuFT#K#|@V4EY5;aU&uxl&-8?IW(R7kyX()7^i|LLkAPU<u|7v1En{hoJO58!`<wP$9l{PgM=1Dttod03#x;"
        "@BW=A*|B6tl0h>Cr8cp03j<5eOuvOu&=U8_Q+`W9`e*~wAqr1-)n-H<Cm2p(6}VO0gCS=`rmhw=9@8&q;nUBJ{VS<wDP<E}Omi6y"
        "EAHFf<~{jrGRz*zUzpmTJ!!$6lk9js&LY0@%myLHE)SfdiQQ3R4#uNGvRk~cgwfhX@^0~%^0A40oY}EzmbTtRj}#O1#amZu)x>V)"
        "xndgG!=Y=_{a?1<d=aMm<%`g6KHQB(0{pj-d!@T}UR!(AS*G~+n=MB94+G`}WE>U^NH_rweqf$8&XJ*xOw8T0TM;MUVje|TVu)Fn"
        "J~O1;kc&5}p6eb1BgB>+OcuceM!eLKgkXvjZWb{sArS5O2q?qr+WZY<(W{&<&C@tUIyMGt?^I)W&d?zi-RphnC8nJJT%22r!Sd3^"
        "Tyj_ZNBN8KVQ=&`p1=r!zjrV~U@ILzbVZJLwD8<+oHcCfNBA7iD*B$E){$)nHp|RO$BG9<iQ-Vsp+T`-kDxs?o0ey`f<q0b2O+iL"
        "b!D5isI@(RPmD_595mL^<>~`sx!rv6<wayEp8+Xlz>?j6xTSS(7k_^1d8y?bU%Nl4<NSl2W)u*1#;7DH)pqj3YJ(ZKiJz0?XYo%U"
        "hpmc6bVcx7HfzcCe(a1O^g8fmHX{7LzxiJW*GSA2oQ<&3hxE!J3&3laR{%LO0(RUNp?warpN7wQL^k@bp4*$>0Z>Z=1QY-O00;m8"
        "03iT%;)EM10RRBx0ssIb0001Ua$_%IX=ZO@VQgn*ZEayMZEs{{Y;!MPUukY>bYEXCaCv=^O>V;=5QX=ad<Voj$_=_mvdF4Zm3CKE"
        "2*bpt0tOA^roMf#Q#<`p%m!)R<73{8lydLRXtX~<2wG?|ontWEllCH@?LFj;Fg+1O_Jd%^3|3qmcOnmwJ78x@HD;u|l~UGKU7ZPo"
        "Xs+CL<ZcTIeBL(XYzxALgv8>g3+h0a^NtIF9mf4+SRt5lsg`u=R3dyYog&S=-Sf+FRGwyGg#)1hYh7&f{932v2X#TRY7tdJ@652m"
        "2|z7arRM-M?F45iBQx+g8J2)6sHoX}EW>Gf=dHTd6uUU6$+q&IurI#$)<MDpMuty56$NM`rrVCfUbE@ORVk&tS4upH*SZqr$U7g)"
        "&9{DcYm4Jv_QkL=>c#VKnLqG<g58wg`NsNNU3~#iO9KQH000080000X063DA*mDN}06rH003iSX0CRF<FJftCZ)0I>XJu_|VJ~fO"
        "WMyn~FJf|OWM^eAaCxm*TW{Mo6n+no|3G9fQY$N$wR`a30lj4{x^-QeVFQXnpe4%YtXo9MN&2wgzVDEdOwn@IZZm?`A`icFcrM>L"
        "^2y1`FI6MUf@e&Lir2gm$&5FRC}@?{A{DhLV!_g~W_+1(T2=uo%7!&-!4i@3HftEK<F#x=+_bd_kB^Rzu4z-&Sebpeyk-rrSE6C9"
        "5{l&l-xIdelGbI;WFZ^LGx=GoV4e~HC}VZ{?CQnqH`lLTT>p0UD$EmK!@udJfwy7{fbnAOorPyl{eY<$JH<*H7}upzqZ=uiw7Bl_"
        "o{@NHOqE5|HoHx!SX;zvUck`rfrIC`6lp4B2^`uc2XI!2I>rKD309`8!g*0|MRI(Uh`3Bd&6Z-#w-Ua?AN;7-Ry9uuMuqH3K-rHX"
        "&p9k;1a`EM9lp$~OyrOh3Cuh>IiZ_lv8)@G^JaZ~q&?T%S9IHMqmQR0+mwjT_$e$3#-XLyTMkp)yh{m4ToTUlQJit5*wxP;-j#Lk"
        "LF(r+X*1!^kB=BWunl`Ky3ThYM6eTbvv%LImdQ$d-}!ATYMDxGo4aTH9)77smGM|#IP5M<*hn!JMU);<gsjNY0Ev~)$zcH#m723O"
        "D|v$?I2W_;dW-0zTExfm!{M}V?ExsnX&l49&ibv8t2H6wqVG`@xo7}nJX97!c`(5DZ39srx4cd|`|jqQJFi8d%GzhMA4Ut7jayzz"
        "P6f7LcVSiDdKZ3}3to7$^MGxHsGw?bUAMxry%s35g59x49cry^>c$(O9uK$kA!6Ogj^&S3>My7|p7q7?T*qiKx>}lP@!25H%mI`}"
        "QJ~N%37<<-^d^5^bs5hKoaL>;(o0hqVeh<W^-5Xp=;|iY5q>R`L=+Tlc#ZU-#&k^^sJ_NL5w60ZOQk35)syGM5j|p8bq^y}by<~^"
        "u6>QVm=`0=qRpeV;0bc?U7If<51bvx=q*c}o{j#OgJNW+UJu!ul)Xbz1wEH41l9HfJ3ZYgv``bWxozfb7msd<uCwGznaSqfI?yp}"
        "v_dSSEzjnxV^K!r$!M9SJVr`|2$SJ;R4%b3^DKg8f9bWevnPJAL)XP)5if_QlO3Z{3h*o|Z)tyEKq}%AoKSogb=h$QS^uX=A^W=Y"
        "xh5t_N@3`6)uJ>(j?l$Yc?WF==y=4!JNDgSgQ=pMO+@rHXTPrn?5`;bysEPMp7j_U{HwmlTL7ZqsXsL{SevwQ6Aq;^$y;NB2@WJY"
        "QQCpb9dB6)hz+8??}Dj(>zlzxNPU8sPPB;>>KPPq4IWAuOYUiKz<<lK%pgLX)rn~SBh_ES&2|MWY&W?aUf6QhlOs2{jA*_8?#T3x"
        "gUIkzrq*|pcytveG9?a{se>!QkG+MCF^|TYor6?R)EdsiGj^(beyFK_bjJ0Z6%8D<p8CwB&o}J-uFM>On}|x#w7;KZA$aXs7Y&>{"
        "1#Bs~T3m8C#5HBJk7PyYYmm188xj(JAIm*Hy6*_fcHysYXYaIMhl)w}LNIqI(m1#aPCrl`il^nsFFZkM`JhVN`AtIoqPh1T)Z_>{"
        "UPQMmqQ2i~;g{yY>lPhX(E!1!ewLc=#(mJUYU4F}2AD-R8KN5>m@fU;aHbvI`I?KKqKdM@^Ug8(VCj_M+YI0QF_lHDL0VT*R<3Xc"
        "mGyW}l>x!lY=6tmC*3zEN5w`Dd)`N*{1dVYqq<m_O--fWUQCaX|JW{61FdR2W=^z@FBHqU+E~rPmUg<vL!&Hu-=(o1fyRVMuZ-^N"
        "muc~#og2;sICV>-ih8<Q^GYyp$(wkst2D|)F)CNznHgMH9KwQ}!Ln%O5Z2gVy5nJFr|AMy57K2T7>VpFIoE%kDyRDB0|dt&pWR{@"
        "5xSk8o+BF*8Pi4Z;*iM%q1XsLChf4U^xJ{!QZvA}RigjzN9W!tU*}dm<n#|qOFptXwSf*b@=iLk935HxdMv$@zv(-s=}F!g-SXh3"
        "7c?!$8v|Q{+g@;cK+vX{D`7l?S+IA`(7LJT+?bn8@Gu#E6K<ur^=uD~yMHil{(jhCpjGrH;vr2goa=1{xWR@07LJ^-{qJzLtDfmu"
        "`CoKIE5x+x-8gk&MON=`T*U*qs2D@JHO%N9Wfp;9SPRUbx0!c?muu@j^s+qz7x&|sp1W){xQ1Q)_9W<B$B_)-IS*}W!7k?e_5Ka}"
        "FhxR|Wxtd&Ms_Vmk00kj_ikzaoqzfVTd6;ht;ScxYZFF~-D*9B1R|0sm09NfWg`dX&A~gF$s#@Rom))IED!2I!?wZgIB6;KV+S3&"
        "3{{ynIluF)<NEib#q8Wacx-RN@oWJn|9;o2dW^x_&jd{j#Xmt48rO_aqx`h>1#EkOmX<u;*xWQ|8YbCkRsC1DqQkgEC9T@BRd_X<"
        "Ca)>z?u{LxH~6DD+v~MWOoY}C1pSXlWie-;f$;~TY>k7DSpQp9;r$V?*OqoYndzTr2QTV_!*mAbDxJ<@Qhon7wXqaTu)rUU?|FRm"
        "H&9Ch1QY-O00;m803iSgX`>3i0RR9)0{{RZ0001Ua$_%IX=ZO@VQgn*ZEayMZEs{{Y;!MTVPj}zE^v8`Q%i5#Fbuv2$bTSwYJtYt"
        "X$K#M4MPhILwCr6T@6K|V<ECAkW4!7*N={yr7JpMx(F5@--C2jRgdr9H&B?u(QBj(-uWKo4nh=-*(-ER189#xDL_o=D9TRb2|+M*"
        "tFJ!*Epu>6Tvt`KTr3yolHk<&<U~*)4ux2+PgBzKo0a+gPth)R<Xzh1Ft}R0>$)@Qoce%HkBr|)H|WB6&rCd3E*BqPbUt5jpFIWS"
        "M;wU9X1M@6p0eUkZM&HF$c1Cp229Ny#krz2`TkJIf-U1Iikb2vE8zcR^)Up@t*BF!X4Y75G0TPUOmc&TBwBR*Jh|RnAUfEiw#4ns"
        ">1Hze?w|B;VfEonz0<LYJRB$2Am$F)8rmRp8XGv^Xf033HXe~9t|hx7J?UEMfI0p`UO!>n_bTg8-?vaGML*UP`%Ci;u4abk{lKiz"
        "aGq&`>l7I<M3ox0>|!?F^Ll~vOs1cTEzSgVZ7GUHz~x{pan(~X-Y1u$4S3p4X8@Vhu-q<TfZb@&QO=<3yw-myzkV3J;F}9{GyMV8"
        "9aLY2l5j8)uMmyO<o|l6Z0k->*-YzWsjJ)N;x|xB0|XQR000O8001EX$VPJ(RRjP4r3(N6E&u=kb8=%ZVrgb?V_|G(Wo>O?FKusR"
        "Wo&aVV`yb#YjAIAZggLCa$#<BW@U0NaCwbZ-;d)o5Pq-J|6wpM3D*tqz)M7Vfde4{TIsa+(p^PvGHFc{JF?yGEldA*W*nzY+8%p;"
        "i83C~&u>0k2=T#NReJFF$=?~>ynvH6deqKX2i{8UYP3)r3x12xStOa#Sb?<VR(UM_-m=`RF`LikvyZY-mGtO9HVy3X#eM6xvZMbs"
        "k`)B0U!XMIO9p=#(CD9y-9bsp4wNP*%0X+YeZ%I25NuPmov|J)&S$kXEp*av8?}Zgzo*}knl}_#)Ep-hw^>w}4mBT{&nm1>X%<|C"
        "qTFJ+>x|Mqna_aMT9&)D(YOeX^Wbj*-z>oGR98^^mCIpmOan{!RW^<lE!K_NY<<Bso$k(ODct=86_q{j9k90gMEim=$0lo}-t6TD"
        "i~d*iT3*@>deR!=$(*M`7i=g=dma4LQAemdOL*-iyr>H~9AvY{vy3}R@<lHXr4RNq>S^l+d%RQ)+IUFL*-{THJ`76=8QV2hg*Q7Y"
        "bM}m!w1}sV{pVTv?YZoU^ETl^rVFU8Y>@%56rC3fsPLdlT#B;Wixi}zmdgrlehx++q-x~4LCVmfPkM!nah9Z`pj+jf(i_rR#p=Eb"
        "5y!MS-&3t9!HEuMOZz8RPm@|i%cGch(NH*qk6g?5)*74C;?Eeu8BY+Qn8RzdvdVvWI*RoF+A@SWH0Hu_7%oaP7J@ohxQ2?Dj8jT`"
        "IjfF48#D6zuE8YirtsA={)YgvUV7p4X2RDhAs_J}1W5U|(wEzi;46i6%)yzxEfHRWkK<wh`0KU*qqw$yjnL`P1&|WiL_^9IEqWRJ"
        "U)MHB7(s8fRs61%SHyAuAfeTHrKML!gCo|Vi5*C8-1A5-*|^PWqe-peme??@s}aa}(7JwxPfffzW=GC^G?qy05~kt;P<^3`i2D<+"
        "^D88970=4dFV;IA--xCZuB^piZy=;Jn#wcg9A1GoMmAOrQNmz7f4U0kVpzY}Pmea9i>%%^P58ZmQ7anxg4R7|$z$~{easv-bs@di"
        "B-s^oul@b+q5td(>RC&F{2|TQgawB<HGDB*y$6bg&`p+Ya=uWacQP_zubBIdctal7V#YdOSYdnh{Zk4*0vEP7pWZ&uZ>}cip3Vi}"
        "k=Y$0M2nz&k>-tNOo5$A!iKIArl6@VL-k8or4|pmm0=wCa*hm|_Y@-D;(!Ye;UWspgG`%ZV`X(kr@T%aJRb13nRr-{)+UtPhdt%w"
        "+o!*VUz8;`@bmq{T}!Elx56k8eml_&8^!l9O*_G#TS?RjzuiB;{U7gRk+RzkMz%7xRV01DF&=v0?X4ca9z=K*^`VPAUX_c<ez?^N"
        "vZk;imG6ctb;QPjn;4Qq71LWL4)OKKRz9mdwH!k_3&yIbOL{`to*#?IRdnJ^nq%!Lcp3aLE<QnTN34>m$|NjJJj9h6a*3*ZcHbrp"
        "&d+e_=uL9s=D{*Oq5cC<O9KQH000080000X0Qo1z8*l>v0D}nt051Rl0CRF<FJftCZ)0I>XJu_|VJ~fOWMyn~FJo_HWnX7<VQ^?)"
        "aB^>IWn*-2axQRrl~rwT(=ZVJK575qR3wy$G;}Z~R8%AwFpyv%*f)eCH@UWk#16It>%Zsx(!>b`M7|`p@1A>nch6-un?3t<19#z4"
        "RtTSr>Ppiogbzx7#nNyo;GL|x8et<9yp>`*zm&q@t??e0vS}6WP-rsEPft!yf~N-EwQbF@0vkYBW3z7&-n`JjL<M>;47-I2TeAZ$"
        "jHF$_W{p+F#TInSN+iG=g*eyT7*s$}6?3FA>xZI3QSJe&E$YmHUSZQPi-udp9D7EW5z+u}+8P_AlZsGhvzg_(QL?eDY(O%Q%E0^9"
        "y0O|Ghb<#jxf`!?LC6|}ux6(xWzDp<_q<Z9-GvNDl?ajh?4`@o6QGCU{v$vCl^hTHOl@_*w~y$W!on2}ZVA^N`6-YasJJ#<lx7++"
        "@p*_c{X>>)-PKI((|kZ~*lpn`E#V!z<xSUkqEwR7xd*~T*Kv(p-MXlFvy5Ad*1mI|fJU2IID7Q)!nNAV2hV|C4<21C{QCTPT-OM?"
        "QnHnuS;7nd0j-j2w&pcA`@B#4Az2k}peVTDrYMpDL}R_l{+vUK3@b|}JDLrDd~u!~y>M_|oIlQvu-?>?+4Dud2*)YRpTe~i7!{S8"
        "1yz#fJ@s_VJs&eByz$1wy5~-Y{rAMCzX93w--k5X=MG#lDu`6PWc5VleXmYw9vBtQ!txaP%Y$(;CY=p&({bR}92FxSy2$<_gE|sj"
        "$`<bnwB5di8h>CtO_%dI&Y{o8pToc90-3UncIf0%HfySk)uxxOQ?ca%8)SB*PD@~&eQ1xC^ueOI8LT<et5>YnIK4v6_NXXZstGW5"
        "eND>|Qma>IIKTMq{y{6wGYBUMn`x{S(%w&|s=r7jl8K*<H>7E);$FAeq~pVE{T$nUAr7|L<K~agBN32{?f<4T)ZjY&szT@W)z@xX"
        "C~PH-GO6}d8QYWc8-e9<6(>`3dKDksPR;SJi#GI*(IQzhQ&Nv;{9e>3;yHQj&R_<>(3rk&{{=e0enh&jn34|Tl=?VI$smRy?9m8u"
        "kLROI4~s(sQ=yJ&I2qD5k_q30Bmmd$7d#nh_MsnV)7tRfa1|==wiCaZ!6KUutvQ^fW1xs*|9S5>sVC!jBf4#$$JHRv<N%ib#c&W}"
        "(w{~4%9U*-yOG^5P)h>@6aWAK2mk;8Apq*<Bm2k#001Qi001Qb0047xV=rQ9W^ZF*Y-eR{ZDB8MZ)9a`b1!9WVr67xX>Mn8E^v93"
        "R8eo!Fc5yP)c<hgg%-8h#ADGAXjB4eL#XR`L8x-=>tb;1aCTO*U(Y#7)1_-S(U(-d^LO9(-T9Ivxjuh814+*NsdRbaEZP8DM+!ov"
        "?cF>0b~OhdMCHL_AxH#F;RX?ulm6`E;_~yC<@|DayO^iBN|PiR4u*pbxg3-TLK@*cdeDqaVSf~p+OyzDW~sFxJXjk8!@-9xal}v$"
        "XfKvj;&?cy&tRUfQ7LBk%^5AL1^Uv2*>C{-3mVl5t#k@$rLg|aVC|e~;$GZ8N{E7T3`l+k+J-hI6T~^*l_YV-;;CV!a08+!Otp&s"
        "tZ>I-Rn#;_&FvO?#$wWr=%k5td``z!CSPSJRGo9BEedKA-b^}$#|3hH4Ab||+QSW6YJ^TVxb-iT!NVFiju51?thbbbuS3WXToH7x"
        "ACS^$L4^3!)CBG81Yu?7D1~Ll2g-4?tqYN4JvokI?+xa0tYn7rcj4G*5Jn5%;1PPB0THybDO?I;7&|Y0fQ>W8ZDIbmK++N5ra<|c"
        "=dj9|J83y})>P2ApLY71KYDZMXu3sxmxYwuTOho;*u{J2txYvAp*_NPVM?46IT|H9<yY4b0UMSX4;1{kcT%iNBdD6}aAB)e$bLeN"
        "n<nFvyw+?*bTj()l^@`<92U4i;`U>o#Tp^POgDtRTE~WE56=m0@PExYVUQ*srbp|8oBOBaLl68F2KF}b+$hMjQfM2aI@T?*QNP#$"
        "Plfkd9wXWW`geH<53h&nsLc9&e|h|(;+uL`K(`Bg(~gqU;^XlA%An;M1=sjJrS(37i$j98%+ZX-Pdl_5ls<7l!xhi>um_h8g|@4T"
        "FU_7(Ie%tAV);JQix%D5xRcvw$^#)$$2x*Hz9Amqs$ICdN@u(G(=Dmau;VZKggU}dl0Ci+M@Qbpq%mLu$*znJvG6}oO9KQH00008"
        "0000X0JH>t$`S(r00IaA04V?f0CRF<FJftCZ)0I>XJu_|VJ~fOWMyn~FKKRRbZKF1X?kIFX>V>WaCxOwOK;mS488}*f8cydfz;_?"
        "yJYAvU`U6eYms$DuZAMgi8fg>D5YJ7{rFLSCri2l^T{BokFSqpk|ehmZ|5L4D1Ver+W;lGC$z#!1Ys5A+pr}BD}sJPct*Y61PCOo"
        "&ymt3Nyek`C?{8d5<y5KyhjhZC>@1f6pb2La3nXWwIDoL+r3`t+(}{9SQH|a&gOdEqAY%vNV-59zLhAC*?TGS+!@urxIz)5zH{Lo"
        "h1xBM97Rv#(Z?>N3B&wCd-*_HJR6TH8C+W(v@rS?r|ay}2s<B-fIsYFn1c`02(5GqDO)+U%e>o?#7Yal;m%pyH+puY{6DQqgC}hi"
        ");0rut?(N`s45Q`H*VwT1qbB5qm0=N07M9!8OyRDXpL=-`vv=}MyNhHuD9i<z9|#);cAaCg;QI@gec9{$08an^^jvz;9mP}feKbT"
        "$c1Ej?BMp|T}!P&>~rQzdRqY=tII%rI_2XCeLV9c6sQZKFlSF*l3rn^_lzc<c{xAe=^31Vc!5uL*h{#C2$43@l&gj&)6D05PXTB="
        "o@7Le(nvd$rpJC)6c|X$18<;t5M8KXtC8Oe#dDwG%>i;0)~JNsZF?loNv*Hw8J}8F;Ix<R=RZ&!q|_E5Xw2F-tw}XebxBUbz0%Kr"
        "2M~)l07<x@nJ6RO#mZpPYYscg#qaxBMI7NKiN)v4pdibG-4i&8?+2Z%j1#Tm^oCt>Ha)0%KiH8BjrOc3bjM?4Ck!cc^e4Nqeg(5%"
        "+5+=_<W1bxGgpRlZQC<qoKV`Xr|q)%I=h)YRup7lJ`V*~ko9T12bP2$i48x;t<T)^T5TECIx1%OZn>B(zbs}qkLPhA^24}WI42o="
        "3J=Z6YWoY<58doH#pSbQgM+X^$h8#)3gl(T!SKP-fo<YmQ=0hC?B4UtKtREllcJv|B+nRW4SLT`D5hb68)<ffLMtc*@h}8_P>)-+"
        "y&;lmu!z#Gl+B<C&8<Fm*e3CWCsRo7>M9Oyd3&c}BkCDnAzI*2b*@pR2{YjjP)h>@6aWAK2mk;8Appjtk{!ne004^?001ih0047x"
        "V=rQ9W^ZF*Y-eR{ZDB8MZ)9a`b1!XWa%W{^Utx4~Wo~q7Z*DGddCgeuZ{su=|2}E|!$T1$fi&ArfY6{w6AU=+Hr;JwU#6*Y6L&RD"
        "?9_I;>k09<^Vm)rJ58^55+A5Et%;x4-}CB@Mx(3Kvvc5eNhQlj0r>xk=3}BY<(df&lFq55JfjJ4p@G0}v@8h}WJ4t$Rt3puiL~kQ"
        "(ecqAf^iK>%ymiL0pU5Q1(7s|S10#SN%0rWj8Fym5?EOkmX+pdBlSkSxlG{A>&xW$h!f4$6kgqrjSR-0PBM}$sIp@5OSY`6HZEjI"
        "3ih5@b4yZHNQ;)0gmJh!{pRHA^jrM<4h_7e6mpTNlZ%_5e}8#*`SabMH<xLd!(2$v3kvid$+R&vudSh=3BsYR%7T?lTO|rbjYcEm"
        "F!~^*1}$W^2%l2UfhgeIkR2c8bPltc75VfpWA${+6q~Vv=}mNe1o%OSMNZ{(jlSgPjf^|Wm5_;}$mt;pMT`0LnekjapM$~XZ@3nm"
        "!o>bVq8P?M1&rs85xsduH!ds>jh>`6>G6kFHDsh#q&RUZXuc(Zr$Wp~QEb5aRnXMpV??ug=vvBiR>Ef!cmW7{;8S<hm}NeaFh${>"
        "6g9n+Qpjj@Z)g!XjI~-SC~Ku*K`>PwW<+O;1Yua!CSxB>_El(0yvG!K{BZhcbcU9+Cm%HAvT@#IM~jd(0Tg?WiJG%H){d{<3lK16"
        "3(5w_R*HA_;{;AWsQMO(y&`ScEEF>el~BwS&sKFo`IBD2u$O0#j>vJD>YOw=LL}i&G;-VtGOt-iC!q}YveDCJHOkn^d$8FD=a0s("
        "?(KPEoqs%pjt7J%|I0xIKMlx-u`k)HlqOfv)K2b$WBIa&)J%vXid&IJm6MZ6p_C^m%&%Rpl)kGl*wOebffm;BS%aRWx|STG&R-XX"
        "jOZxxElxtcF@}bf6O_}#)O-$l5^%(>TV(}N3U1oge`SZlMVoFB=jj!;NYvc*k6UcQMK>*AR)}ngiHT|V+uIA*K`D_OhXGNWvZNXZ"
        "4lG!nQ|^t1v}vra;mu(>Wt{2h)C0vJk#vtO<OqRndfrj%`M?oQyy=3HT)9R$f$w{V%XN5u=AD~^t<WCt!5(a3_HtU)6_rt(Zas~="
        "Gc1ad@yK&A@tsOKH<G~UWNTvF?<K?iWE)$PNjMp78`#@U8q>DQ;foNmL)zZfRBphrwn}o=R?%#OL*JbS8V$0k2ZdK|_|&8GQ1aT|"
        "E?wutr1NKhruD5MYae@*E$T?se(h2CJ#Mnwmv;Q@Dg()*5Lhh3m$?ZWE)j8f<c&IO&K?NNhZYze6!n^Mtj`bR+Kdr3`IR8Kr>q~J"
        "m9^NQw3RrVu<+FuqCKLmyyZv8+s5YOM6UmPk)qbSLWf;F3~7U-*=wD=o2%Pve26+Bv{I8ZI{t1Ktxx3o9R9?n14wIk;I%1I>>jpl"
        "a{(G_mlIQz3NHt^vg1a)z|p*I^Jzc5Eg5#p^wjU-DEpbne18*fQBk;IQ#H8U+Fa$IM7DoZ_I2*3+P5y7PqS;HkSPxG1g2h_apt-8"
        "D(INK0uJ2?)=?`yr1sY3fMI-QIr?pHhj~z&&pg;o<AdG&*2r_KHtxK_?ro)AgUP^D=?@WI<1H{6*vn&MS9ls&e*WSyHf}kR(8z%`"
        "*Yl_oJcy?ee9-BkONevY+#_+OR|UgW#|myXy&cz^+%etx@Pz)QxnQ)H8>cc8T$_6?(Pq2C7iQJaVo7<R%#hJ{8YcUOOAlg?6}P(P"
        "<@Q=&_t6l#q*c-EqGz6j)$Zi-Mnz9hZ;qvFMxUd8t%Z|231@F#0`@SjaD|}}?%XeiN7s^WTqHbqd$%J?w#O4Uq@s@ZLtZqxlHJ|n"
        "KX7Ed$HMjh#==#bgl@t`t2o5hn`6<k@7X$?7Xq^{#*ylBQL2^H-TPFnY9c9qDJkLhnzf{~GAVf{YwRgsgPnkJss(c8dQ<tM_eqMn"
        "Qmqx;06#g|#UeC&nMztwqH&4WRAMczY2>8s7KCwS{R#9KHS=o=a_#vADd?U_*!_~7z0+SBG9kUf#s*f_MYL+sMDVu2cCj^}j$gzk"
        "2Hngbkm<gpra{ttI%h=@{nNOTfD=0!Gd>^1!9sGe7A%Kra@}qXaB=+$&ckeJALe$FX(nmmnlA!V5xhm_u(H<*DV|lj*Tv@rE-q1H"
        "MoRc*H%F3&CuvyTQ@m_X3$~<@JMitrvYHVa2If26`~-@^<2?YJ?nyyAGjkj5ASi*N1lGeWz1nY2_E)B@ThqJQQ!Q%Mzejvb|84VN"
        "peuM^n;$oVbvbJ7L%M$fP)h>@6aWAK2mk;8ApjneqI9tW004sq001BW0047xV=rQ9W^ZF*Y-eR{ZDB8MZ)9a`b1!XgWMynFaCx0o"
        "+iuf95Pgr7e;D~Gk&LMj7byrHDq4!v6hsj`AY`q*V`s#A*WF#WHR9u$^(A%Ev;r)}@ntS&&Y2lUQM5jKv4Ywf+2o##hC<}t*cO~J"
        "K7o_lT2W3OwD5F|;7PYYRjsf>?M3H74%SDnp7s_=-6l~K&1dr&n;GkYqvf03yF_ac4zvztg*6pqRG3_-601rixzPocFnBNov=q`^"
        ")|fwF?=8c2qs!VzR395(H$H|-yn4pJW8njg``#`WMPZa2oljBt#^QzXpHay6#9Ctm&iO1?!Z|qZo!<xNF402kgfAL}%hh}ad`K)H"
        "O-Yka(}hD-#NZhFYOj{jJRGj!H;ZMmbz4r}afBoUfHz&7fMraBI%f=Npa5_Ekc>1}o|1H{f$uM<ov|?p?>RS%mR7?3+<m)*CvSHi"
        "Owh5Sr%hmG?1MAT3*V39bN-Lt@#)8F&<4uDqZhVB55|IQwWuf`Kw8K}zQMjd#>iHX!-JeJLTg4bMBj+>`pq_XMv|X&-73@=JBgN4"
        "S|oO~C{vPmi(M+?yW59|y0hp5*#mq{j0Oy;<B7n%SnwMzjK%}j_e1X5{+%F=?pT#jA=55Lz@0dJ9e-tINawa6Sk-F)87iYo$KiQ)"
        "U;dvYgB(NNZEMBw>HGwQmT<m?4g2m%Qy6<AtQ=Z33uC$38Pcxq$p{}Rd-Z5DgaAT{+B2@RaI?n^KiOjY$I4F|rJ$SrZ8RlXKHrcf"
        "U4~`oJh6?%tA>${)z4?4_N*Z#)7m2ME7B;_s#_ojmqS2#Z`TZ5+{r(z>x3Hz!fu-cXQ>$nch!cZL=Dp1Xk~;1yd}rWa)Wj<jE~el"
        ";eUPfJh1R#`~q}3Uo-5Y9Tf@ti-e+I{nsHw7M!~zpy#FUu*wvFB8C>9&;9^VO9KQH000080000X0BuGL=>QJ^0HiYj03iSX0CRF<"
        "FJftCZ)0I>XJu_|VJ~fOWMyn~FK%IUX?A5UaCxO0+iu)8_B}xUft`Jk0$H`)v`t;8*ljmmESj`I6ZFL}1dYU*QLS4<$@aA9e*2zt"
        "NJ=C{joo$tXGS7<c<whco6Y9OY$)$I`wy?{jJ;$-*&JEf41C|iK-*lgK5L4$X7{|@9|jRIp6z+hvSHv2PKa33vVqrK+h-Nav;4s0"
        "=g*!$`*7fF?8MOXtcLN2fr%l@Z&`ca2=?yH>xlJSl*O1;0Q2pecMP^+Kez8*$Ly!3I<kW2ZDDTo{8=erO;=?(FJksStNGck?eDX`"
        "VB(PVykI~40;uQJSa89%?Qmd%S33fn_pN|;Knvg}0Kv^>gIJb`_rQJ?2!GcDo$^gpZsVE{S&<Eydd2Q|FG|lmpral89QITbcV)$e"
        "nl>D}vf1mAJQIVl?a=o5!G4OHhGl{^dYb`I7C>eVn`<Z}tCt^ur{A^BuG~k)o43HU%Ck1EaX`d-nkk<Eu~a^5$ieex3_dfse@$X`"
        "NB%A4^Cmm;UJYhlR~^&79kTwIcCAQxy;VaXH=YgY?HxRa$N@rdTkhICs|bfUCt5TnuOB+zmo;quzT-J)2qOHx{Ee?H1zPiA;t<XL"
        "X&kz7h}e7n@e=;N&4&CCu@CJnZvam5Gw-spfBlfvUFC&}vIo#rm4Re5WVz})_13pcF&db@=S4cHuLiIk%c@9!ZNbLUyHb?fvMPt8"
        "4-k-4HXx(8-TumRw9<*b4rxhAsZ#*ox7`t$`uthJcdQ|cP9;AB0kmGt?2Fioh>ZLSETpYi!anpPyyXpvb9(zd*d-6Ir1Zd8z8|ZK"
        "6(|t20l~oD7^oIQBLSlm9byPnwqUzovmFSd?T;+0?z5v{qX5)#)0oT-L6$phPr=62(@n@eN!aG|_~NVhe53X!4a!Qv+0P>|v*xe+"
        "zU>3;y9{^5z(?F=RSB{Ke~|5ggJQ4@AxR|ccoQnE6@m{yF<EV}u|z;)fK!a}0;V5=-9}nW`a3c<uG)Lv2jL$vdJqi}5Ey~rgxm^D"
        "$qEq=Bybg3k)ksUh~FUGWHS!Cvsariyx!n|65HlxViO5#=EHqifMZM8XO}pLNU#JL_WJVM4FkR7?B%O@vcx1FJLE?&zo#)EbDJW7"
        "<l^#Gv^JQ69i|mpc?v8T2hZd~SrojHpa?x@9OVPNFG@f{&U(pSFyBlA(qiA2?p)+0Xk_A|v#;81R;APU#TUi|OGKJ6pr0v4u7uP+"
        "LWNDYWl{E0s{z_%$H^wEstxj9nT+K6k8Q&hY~{$3g`bYm#sQ^#2;~0Of*AK`(h}Vf`aP-7AX*Bz8|_Djv2WB0Y4Ujmy2Yd^yd|eI"
        "IqksX$PJr*2a${SU_jDifLG}usDiBEMGXp!ffGV1L~O8|uW{2HxCN6r%XT=8!YjpW6;2T_x*!#o4UNO8h8VP<fl4ln_YgHo#1caX"
        "FVH~D(MbqtorzmxK*oOvFCK^xa?@LWOeM<`6qGUi!uYKOnp()Y3-l!NGVY*CTKt1AUfcpxCW1o4s6m??JkgOg?Lc~UW$U-}gumc4"
        "+YeIiCgaY`Dps^yf)ZmV^9(y#(_TyA;bj71gP1Vz_!a}q0gHG&@EGp)Fe&C+mx7*u9KjR?;}2a`=H)P%p27|lm3D<l%1|{#ys-fr"
        "G;U=cQ1NCD{)eStViv72C{xER-Ri){vcOq*b`g<Q$J)^td*m%7wW37x^=R}w@8ifJN+uG{$FzYE6{z1`-~Jj{h=uW8$?pTgCk&TB"
        "eCby}gVBT#xVA>XI}Q#OedA>qQ*eaoJOF|&ZDk-_N<oGtO^iIUBUI4ZNP1Kx#WBTef*~C_B*Xm1FePxD4k!$hprW+qj(A`i!eU#^"
        "x?6*Fw||0lGZ-)Ph_JuHI{Zuwk+jSR-aU%oX}~h=o78e?K~hnd@4coi)&2V7#<PIXRnch<$MS_K*RpuHiJP&W{TD@Fu?A9GOtfmG"
        "Go7cU>?KJt)8LaUb2;>wg#IFMiLtEaAYcYcjWji`aKng6b@Ho^0f$~iCh|{!rL9;1AfE0ZjZdWq0~2+O?@`l5b(@Xd*aT7{NQ4U*"
        "_m?+;6*M9{R46wp(0@ZkdS5n0dv5@j{G2LCi2*IKRc_z6V<(J-nuYUH#lfd67J1uK5-$Z%XG1+!+MU{B#7J-k%!i~_QUdh@P~w3l"
        "f5aG~fYDHKRKfog?;^H?N?`~#XJxb7gzK2xA9&9LwJOe<Bkh07d55CMY7VqOu8Gu=-#DmDT9vmvkOL-oQ1xv_Vk%7>MY|j7?16w6"
        "DE&mjA}l{c=hsSZ#58&Wp!K{$`}v9WO%w0NMkV5L0mW`R4vyQi)<mpJBr{1IbJnQVVzP1`p97n8iJj0y{j{hS@B=&<E36X(8^3=l"
        "V{lE8niA!HY)27<*dsz$vCJW?9VoXaR9fkQ<feeAjDBvq{{a#{s_X;c5&Re{vt-v<;naJB9G+6dEq2#-89)Z5bLx7m-=Q>v5_3wn"
        "+HL@ybW{8k%Cd4uQ{piwT*vdPMc2mHYcTU(NJ+7LjwfKI+LI{(7}QDlaz=E2Fman-fz$thrFwEKr~`s1CHPP1Zm09gMOb~p9%`+-"
        "$JAPprs;Tj^v=d&dvGc&y$Y!t0VX$eur}&wKZT@Ts%4Wnd+3OvnXz7pz>L&FL-XJi8mCe5q%)w~|1HphA&!KqMA6$OiVegnnE0k_"
        "c-9Bgj7YT^*tX1stnM8NIFH!Vf&j!Hz(_Ih!2|3!LRE#00*lu)F2buN#&Tss?cQLeWhy6y%f)luldM$r9*IX1=~LjS5hWwL>_HF$"
        "F#@B`Lu?FT1Q~kxC6#uC!g(zw&~u;Ol5$m3E-3xD5|5IJPjlzS(~8V!Q3|d%y>I(M1fk`W=Jz<yay14ru{_LhjfX0<bif9&P*a{{"
        "6;yvt?kYwcv~KG}hj4P!Ez@E=dp`+1gLW!mlsn!vc)pU4z2M2J)0GR%sp|VQyY-L}uS`5#HB<3ndSR(ngcV=8ZxXBDETOA-tI=J+"
        "T|$&!6HNK_r-+feYI6RGdz5+2heKN!oYpnH2r!$mu_iF|9x!h9fOtOEVyq#N3fUREfafZ!yR<Hw;M`JPsR|fVs6#CIM!iocupp_i"
        "yI)TG$kk2Vlv(q%Pi)U@l+QY&ykMINJT>C%fJaoVj%;oVX~KD`Ezwb*yL%bw&IuKd8-5SPcn8jZ>4w~$&Tk-<^gEW-SY+TX=QzR4"
        "*>>78nMLFDFf2JGt4&M%Ox+;_DAjXTE=S<bQnzVsBnc3j#9)=J^WZ-~?ySqG6Q%oS92@lPhfjZr)cGlnQK|}qlnmY<+NzShH7N6x"
        "ipKujb*XKt*AY6&zeQe*DPj;_$8jvv-5XgI%=RG#npW+;9K?ES*(Vn5h8+oLOiy=uNs1@h#>8yW*%YnB5{GG}vil3#_4^-xCD}51"
        "vvyO~lYlrA9}grat<2);->@`d$U3sfXF%1I6`@7dYY=!b@Z{4LI8Wc2;L7gVCyB!;1^Vjp#>3R5b9}9YMqAaR7~mHejnkgC+@qxk"
        "6vv*}r%P;1Pqg>s;e$!8XR>`#v0@*Nz`*zWbZ|<8947y%BMg6>$+HF_Kw84Fls&;$+jT2?{;umj!IqPb+g$f_w~Zg?jEDqq%A12g"
        "9QWgJdg6=xaeakNhbyBAFWsqToJx?j!!}MdvTmi;+?9AfN?=lK8o|uYYN7y+xO!b1CKP0cX~>#Gpx%9oOR=#AUyEoX=XgRyg9qs<"
        "PAt+WOQ_xuu=M<%6whGAG9+2rc6v*Dw`dVfFiW=T|Cw5S$|^~ucztEvc7cTR8}lyA<0<1Z<`CE^rh8JmPnM}2v1wuA-)gI#&hs*;"
        "tuf}zh7BX|n=f3Z)W?Xa7$L=Ew=F!UNxpn;%CA=liyVrbB+TAGp>t}*nx&mm9SKis%>p$&;AEB-lU90|q`HPiC_9UMH;Se)xoeee"
        "fyH4sUo?`ve$@)YURO$Yp8{VmkW|s>S}JKNtlX9+LNPY$$3|}@>J~fM-i~9z@5-ECVMT~{C-QFG(AF4g)qc96-;Ch51m{I?@uj|r"
        "F&5e6ha{*#euXi1x#_A6PMU8q;gg|u>bM48V$YWkX~(nMwB~i&AIai{1}N7M$EI@=KQF1*Zmm)}jLLK!Tt@5y|2~7imtn-{#pmIT"
        "+gy}QvTJPa1fhxmBwB!CUuz3OLm{ti-Spp|3pV1|3<nO?^4Y~Enh>NcjSxEjJYrws))%kf=_`EmMd${{$AMp(z%NYTuZZ;XC16GB"
        "l8Y^w08lSkO}sEJh4q7)0foE}%X4UBFy3bQE%{=#e!5D7BC1L;1hhvyzYlcky6>~11pOspb}>Fu7qY|FqX7r+h{#a`6Ght&+vj-H"
        "l|f-eg_P98Qpk~DAI3Noh6H+#pr?Hm9!=X|rcE7}S_0X}*%nVP@I}bJfpmL)IiQC~Ir~mLLQkS_z&sJBDx}qu+Wu)b{>TI1p-(a9"
        "P>+_4ymO*FETX8fn<9zwGEU0PC-;$%so=#YLv>JS$;Pxwc^i6T)t0DIiTv`EkJRi;r1;nS<H$d2&fNX?)UhVa_3F!mjO<w_6>kC?"
        "Bh}1aLF3$m!vd7k7dI<pl8ZboPskc9y^y(txnNfN1a+^&!&;82<1q|#-Q*`yw93S^P*62cVq@;4BB@lwM1Fa(NU;f$Diuxj%0#{J"
        "ElIaW%qMN^Uw{R3dApAU^f*C!D+Ao#0JrynbIjAk4r39<rW6le@iJeb02^Ve&K`8Unr3OgsRN!@IcMnVUg2rrdeW0%W53q8L8WdZ"
        "B~IRCYwqgBA#B&%+VAVW!*W}%PaY41J`0h>ltxxUbB-XMvWv-MaASB%4O7&A$Vjm-4$$3datpK_>a4#_`2)|#!I`p_#;s>?(`l}1"
        "+$+CFDVG0=z<d<_9aP0bKjs6IcWR-=$D4^<06e<q`ZZE!T}B1l>hOZNz#JEF-0kpxF--TzVftO!j&s$xhg?gYYNv#Z4#h)%T;ykB"
        "G1<MuKwVE?9_a64`;u4lpm_?@U{X&(N0SE&{|``00|XQR000O8001EXt6AZ;e+K{nO&0(FB>(^bb8=%ZVrgb?V_|G(Wo>O?FKusR"
        "Wo&aVaB^jIa$#w1Wn?aJdA(R$kK8mAey`O3V9ZNWrfIu~QWb)<l&(~%lnOy5R8^6aI5TFGIM~i~2gHBp9A6S=COcas#ICeE6Z_oH"
        "cP^onzPuS)Q3*qOX2hP8YR9YVL5kjxR<c_3Tf+L9u$m2qtB1004@C40SG=z{QCu6TfKv7urS$&2`}agQNM*>5>0K+z>HF_m_V@2K"
        "N_J#m29Jn8c?(}7Lf}{J{7uaoZg{Vy(wTi8Xi&AH5iLhTd$kmetXR9{UB|Lc*1Xl(Y)fGj>osT{7=-VI;Wwt3FP`D%9po95qiE~G"
        "JuB>uHBt~GdhlV^@lL8kF(_G1ESzOUc`#g`B_1S%=n%*%**9W4DqgtfK|21{ym7v1k<CKA<o-PZKYHNR#(k`L1qUj!-XB0{j}G!T"
        "vgiKv!9>0n70+ok){KT{<}mQwhAJx?(8N4>>W_@*5BU{qb>Oq(tX5KX#a5L0v+S{dtU0gqCtt(2^tpQiiFGx$Epc&xocm)Gijx<="
        "{&4wdP`qiyc4xjJy(G2#*rWNHNJU<}{gptJYjA7B+O}lXwad|U+evOSF*!X+Z}UnZIXy{<Xx0dX1~m;qZxXPD+fR-?5a7|N_)KV<"
        ">Pd=L`7nl1W?qN|>53vK(MkhSWPf^NX7B9`ZpR<UKv41uj^9XgCC9#gsgzU+{a%_~h(t|nwh@F9(Q;H(Tt+nMMh${*gf<yX7sj4g"
        "aNcDnioIj)$b&{q^HzYRRveNjUU5V$>jU3Ntyq;f6Un#a@mV=<_|_GIKy_>sCVIpHo;o}@yd>^lnjuoiXE|~B3Gs1P@e#&_?U`s<"
        "iNT3DuIptXvQoW$khH~%Maf5wfYAg%0dX_Do-FYC)<L%=kk@rAn<q~TJq}o#c>Sr0qIMXMQ9&ay&{HNgCsyS9E0Bp=Px_U>Wl#=z"
        "rv*5%wapb8{Ii)#RL2dJC|{lIU(fKiHjqq`G7rejTDc@s`6;k5=D(<s&&d#HcjOhOHCJm$j8lESeyfU0KhFw##+&8&skoyH`dSRd"
        "N2zL^Tg8pwMIZ{M+v6j;6&FM|yf<9RxmE|Ui#uQB>VYW+q2fwU#R$SWn|2OnjpBSm3X;d^30=3?a#O&-A_MRinT}L|_=S335cKY%"
        "COjY)4<3B^_|Dv39^(&v)$6foV8VhlcjDgteEB135KFF(1UL@usHdW0f$9NjhpQ#)kiOQ`j`brndu*Hu{_%$v6aNk<?%x_vIyMYg"
        "O>*)V_1Qq5-=3RR%sJ?!fB(lY@e01Do`8P9RQNRPIBow&+{6*e3Z$?%68~IrGpgPlZ^&OW9pyIzSE7Sl0`FUKJs@UtbfbdmS-3{5"
        "!#yL5T2v-U<ICP@aL0d&4}D%1wp7<}=Lp)=Nhm}O>Iicv_MloL?A3VATfGYb<ZzF^wvz+S9~u<hv4KCj<_EH8t*CA0!%W9rykT|*"
        "w2Xr`in9)!v=<sL1fCg<1!7)fhGKe0kKew2S%4U|uH~qCQ85Vh(c#}m)3t6);Wr`}o}iLV8&Tqe7Oft&P$_x%9YOcc6YV=^p`VoW"
        "*ll1+?L>QcgyC|-5IFSUYb7Z;$idsJ-y#qly61{HkG4=grpfe3jeuo!4ZSzFJ6UGu<s`9Gj|lzw<>e2*Wq3l<3>;ls?6}Di<mPlV"
        "&BJFj)Mr=6I@oD?lF;CjELz;4??RsSqJdS>^NmoByC7)dr6Z^LtQK3Kfm6RrW)QfGubw7I<5Y7EBu%p&zj3#dWqo(1X3zr?<elPB"
        "<1XGgrd&E(;*lzZigZGE%s}a-lS`d}w>b64{!j3$)!h@J_7ofqBB`~2#%rR5QYq!N&C2}+Jv+U=npSr|flYH)Was{3GOUsvxLcH>"
        "1%!chL8<q2aY8>#YA>EU-J_~y^>=k|Dqm+64q=#i%oV}Lz;g=<`&~I-(<3a@esPZG_OM~=yZ4M2v!5U@u}$;v3MKIO>?WN)CqLn9"
        "6N?VQa0<{nQo_ChT^>PBG}J#o1p@;kpf>#J_7>cl>^m6ehHTQ9^@oH)o+(ybQbeCcVSpn}Lc(_t>*Kmu3|D6@d<~fqz2ga>CQxf*"
        "<&PX6#+ZD5Ig>!Co-nXl8`613uZD4AC7>uQmL@mB$7<iP3amBwXzE$aCz*SrlI;OgWuejac}|{OJUdz-PWrb4$lBYOd8pm^WC$-2"
        "e0#=0%H}>X;{cu-GTZ#pD1rk<a2}3Nqi7R13ji**iq+9n<<_wv&*5%SQEay2a#<j2hz0I<8cuv9EeU>~czhC2Ck?F``m|R1<aKs?"
        "S?}@ghu{O0i=QHAxbLE$9GD5iyE{|)R4K=3O)0D>FmWePOWVX{pU@5BktZ%+(+tm37bM@8K_zcPwnxEts`LgE2AKw6P4x8ko#&P;"
        "^TWm>!{!Vjxt%$ZTdTkc+InfTXtBi4F*vo@$Ka@!&N6&hD>)nr??V_wlamX6HM?Uh_G3D@`Djp%CuH9zpGTd_Kl%_)dfQf(xHB0("
        "Ebhk77X-(><GJ_xJR$xAP)h>@6aWAK2mk;8ApjLeczT-!001Nq001Na0047xV=rQ9W^ZF*Y-eR{ZDB8MZ)9a`b1!mbW@U0^ZewLG"
        "aCwziTW{Mo6n+no|3K(T+7vZTvNki|0qfun1)6q9g1r=jKufgEMItqlO5zpUZ{InjC{c=<bOsDaJQtqJ$HPxi6kVO3FNxveKok|F"
        "tWO>o-|Y>dTC;jn9mrNHVweV%9cef#m`;h1pg?|Qbxp~J7IzydSQbUm@zL=SuN$chF;W%#@mnSYxFtfM+E&S$Z1`3dwA!JgtdL^M"
        "ccioU?fTQZ_aD|bDOo=@O!1ltbJMWmE0#vl!f0=NB}-O)YE9Fcl-#ht&fx!+nqr@lTY1L>|C^Q9tf5@3A8FlGY|d**4Z5x>3ekJd"
        "&gz$Lb<V8Dy>$k~V`T|N>w}_AQyuaLsmi<;=8dd)ad^g0&D4&SIW-0fz@-=h*)qjM!LF>r&t5*~3W~!8?3T&RpR6$ap7lXBSe3j`"
        "QtKQXDA^paJI6;w1qdLw0E}TLj3^c&gNL?a$@2J!z^`OmlIO62$@5sVYMT-bjap6`my&ykXp?Iw{*rKE$cik^F1;80=@ty)B<sB;"
        "gN$#j*9-!IFrXird<4`5*7$=&H6y>%s%2}Xq>3ZVU;s1hfN5)D_6(5SD$@BoDVg(oPZg)a=qw60$E-5k9%9xOgkJK%OS9Lo{3TkS"
        "y^WUS{46EW`T`#>F8>)6yM6MK*P!jpK8G1)SeQQsgD#*MgHAFvutEB55M~$_bZhOw77q<b!+26afkPuNA6KW-ynP@qx(p(I!DUtF"
        "dqzv=>;o5!s(54_y4`T9SMO-0+0@PS7=gcM4p}^g%F}YCGYX^R=M=hBCL+UG-dEVUSlQmq2%h*(o}#9r2f(H+*DcIx4MQ0Q!`ewd"
        "pm37isJ$1^8(!yI#!4`JpsHM5knhGiB^-U(n$>A`3MIEwVat^^t6SBw+1SDYel4_klt4-r2t2x}ST5j7fw=}#OG`E1-Y27P_0_5O"
        "O!i|*h_bOnw62qSbvYxsyYGblUYa?InOB<F;~H)_ay(5|+S@7@P(4@7>#LiO@DiSj1P8OrVoSPCb`#fjY&p8?5_0lWcoQzCu)^~f"
        "Hu1#+65Cz+H7+n>Ug~6eFjSG>u|w`>Kuz`C>ff<(-&HG}FN{xJ8ww1>M>{$;T}Xay2x;(QcM7td!YZCeh}$%s1J?Ake`98Ys^8;V"
        "icrZNH+}NmxQVBji6^!CzE3*|d?{vHB|=cd&V4#Z1bUeZx}qB(BuV1<0YXr<B3J%I=DG%I70+M&CA;SZT>IMKsECTDjVwG%UU*yr"
        "ef%UC(FXx{v*CpY_$wrs_>jUSX!gK8NdtKxrGcJqd4+@&xf%hMrVFW<c2pFdI^J1R(b6i{F#qDko30*~UqLW46bExiG9zObc4xpG"
        "Rm4_&9!fgTC*!Ek<75$~BnDQn!2db?FOqKI^oNwZ!I#A)y#0tPuaodFz3lwVI|o;b*P-J}`*J#V+R+8uP&-aTHXA3g&+%#Y4r^dz"
        ">*kJKCqi}wRMKf=6(9)N8Ga)mLkrEVF!K%d^sb?b*32-4rLDL&vE3sZ)?*LiouVZNRM{r<@(h>lhf2Q$1T*vt>~W$UMA#JtbZ})O"
        "WfeQzaJ@w$%HkmUBte|n51^_FM#r3)k2c9r7c4PEzlag`={P?j7j{hcx`pvjdoJ6<loYatfT+C{8wGT{fU_X*0}3}L80f3sc~51#"
        "<NsyYC1U+S1;T2tGyzo#AQ)jUKsI1*IPy`b=g<P0?jaPpIg;x=ZBW6sxGyP+GAjy^A8d!ofK`Yf$gz$`>SM-d_fo6`GY6&7HI{Xs"
        "xBvks9o;CMbevj8V%+eZI`5YMw_7{IEEyV*id|^ycmOMq4>)egbLbuwVwjD&+fhRSX=%f?EGtyrKwb7VRd+diWJPP9Xu$`}e^5&U"
        "1QY-O00;m803iT_7_N^!1^@uN5C8xo0001Ua$_%IX=ZO@VQgn*ZEayMZEs{{Y;!MiZ*_EOZf7oVd6ie&Zrer>edkvU)Q3tfE#DF&"
        "P~swSo3uqP4eZn}fk7|Hk+cxWWp|ghm7@RNnI)Gu$w_O#w#1#C>zOmd$z<}gDl9%&D22AtN>#{W1)@ml7i5-I#%f{Z9YTtUN>PKs"
        "Y<ckZ8qipo3#&9N>Gwh|RU)!qG0(+32tKXRAXHXnq!SIK3JPT*Q-TzOlY&?zn^+YHbY13UhB+3NPBoUwNK(9oL?mmR2Ul!TSWD;R"
        "ptnqF@gd2o6akCIH_MMhNVGDhyAis8auUhl9_4Cn%{-V)CPA>&Du+~9kz~Rc(nyXdZM#Rcgx!{$mgevmnb;8iC|R}0LikjX^Fh#@"
        "lH<ub2!fZL4)+%>Rb?@~r06b$P1cJ5=!d+ZCz`TVSThLzAj3QIH)dHN2`Uh#F4w|r^Bk>~i4Xk17Y83Oso1Z%3y&={B9Sc0%0@CZ"
        "i`vEcCoGK81DQ5sQ7+78k6>fxtIE90H<h(kjg2V7NXmRtKlg=i;2b^>P<Ae*SlabBZv{U8l-dUEt`xehz(<dL1Ewm=Oj4q)+F$|J"
        "sWL9@rB+lMI{FVuf5}Lc2~}yW(k3fQf^(Ks89_AiIpSuDHg}N`cLXpqb)x)3#`4le85RqW)E4J(e0KWmN!SXku;`yYJwEBqiYkvX"
        "u|;kC@#*N4Os-L+?fJ9rn4d?f%>B{H)3fd{-co->C3#M86rD^hF@uAjTz7mX)(Bxq?QU-t?d1uxP0lzTl1bOuL@TPuw$!i2R5`a?"
        "RS>c09F|p<^#tqa=0U6bb<G5(rGm9gQ!3yii~w&BuS3rTWG*3!6Al4gtd0b;jA)HXPTisA)XZ!4`rg2L8*51-txmo`5ty~mjzlR="
        "BE43h;~yjiM2-vxQDxlnB-Q8H&=IBQ$V3%OxvI4DtrzYqy)s>5$AIyiZt$;)@_|54S9T3koUi5tqzzN<>B*)~wGQuQHXkT$LH4dz"
        "pqBgy$sK9~&NHNPxr9TS9?Vj;euCnDZGBjNsPdSqL<%YH!f=i0tA~Xz@A!5H+wT4bw+DA3tI#2Lbg2XLZ#5q1Gdaf9Dj6hP(7}Zr"
        "fM4(fC(JXi8B)`edg#45K0AE#@%U`kE6pX^N*AN*Ts1?}$S*u(3V1IsU`nnVsAy1HLWHT7OZfn|w^1J7-p=Zo<qqn*1AR0Ks>C*7"
        "B;=A<Ap=X2a^nzWl@%FWU%q+|L@o|?_X?0OvK?sh7%`-i9fvQdRpTdn;J&uz=KW@FKg(qge((a07+(!``0FqiT1X1{noIUtYo(`?"
        "t07Je<kSujbNdtpu3TTtC$qjiB8O>*r*MFduq}HyuICJ1oPX1w!KsV&9a|y5R|O4zDr3(~hp4(@9wMB<HV)TiE{K8dS+Iv#UF*0{"
        "E;eRY_+bzqef0s-$5Dix%J1OM{p1-7f|M+9(NxX(-Su_dyA09>d0o-{*kY27oMCcr+N=GKzU;Hc=ej-vH%9^PSDweDd5^h6m>wUU"
        "JRybrZ`P0Yqo2=)Cb(VfM#O43tkEX>oLScKCAAJ_2JhFXu?sq`SKAd6G7s)HaF6i+9qEon7wsc<9*x+w^`1o0lY@OG+KX*BM`gm("
        "d)jI`9BZ06vq$qI?zyoQ&K1yE(9`TF<nIB>Cxd%_<H_U6VyrP7-W$g-*qGe?JyDKy^cE}^B3<$yACc^3r=?c9HAq@_^`Ecqdm4G!"
        "i)AV`CiRXQ@yvA_amO>n!R;;=_WI#YCg}I(`de?$>hHRP9qYZ&8(v(xU3O>80}V`>(gGxDsMEeuB7rRQe)OlAA1Q)He45f)bLg&0"
        "y9JKHZ!4vc`kraDV;Mf4!|$}>6e7o7R3_0fCSdbyqkyqUyPcY+;d7Am0;aH@W2rPE&X9KJ%FsY#t>i_kjs}fke`w#jmqn`XnYfIk"
        "&1C$Vxqoyuh^>=JQfAsuL<lLrn4H(gu~PK5R~O>E{x|zt`P*0Os3Dk$h&Hl9uh4eIXL8<slj>mL*o*`odhl=naIc+%SDtX7Vz_x<"
        "u^Qv7b$cNxRBh@#3-UDZ6xnbn*R%F}-cMfpYMC+}`anJ}LRhq3-|X=Fu5=gT&A2c<6JcLN@GnqH0|XQR000O8001EX#sX1W-V^`;"
        "NL2s;Bme*ab8=%ZVrgb?V_|G(Wo>O?FLGsYZ*ysMZ*qBGb98lNc`k5y)jVr+<2I7tQ<eX~<0mqa<i})^iTAX1o5Zs>b(fi`@nq|&"
        "92W~BL5X9ER0z_xGUfk%-3@>QL6DN2>}{P=6N@Ce(P;D=P5Hk6E=}{S;9kXdrO4{C+IymIlD)TsJFH@LRu(gF%?mESd#p&j0>0jH"
        "@5lGQcp@&VtXO+L{`Z@O@B3eV@#PoUc2`!l$Er0T5WKyyVPcbKtM>YnD2w)@6zx~Bhwo`sZoRn7a~|Wg!cz}_ESsX{l^U>PHKNoH"
        "-@_$RK+jfLTE;A2^X--`;<8AywH|l*KkqL;y!rX^?Z@ct-+sQl!Ufh5f7tOV+w!6w!BAH$u3Jhz0DrRT@_}u4IiKOBV&Lnne|=T3"
        "o!FF4tO=_nYW`4b#4#(%B7-53P>>O_ZWP?uI)x$l@(T|>SMPrL@Z)9l;qo7EuHL+Rdo`1{|DM&Fq$_~p>bBf)9(Cy8E9`Efo>!bD"
        "ktmxghB4*6Xm7W}YbI_ZBod#)<8YMZy1eB@_PdlQMRZ;Q{yW%#G~-F6M&M54e+f^$mK!t$N>70Me?i+Gdh@?Jczz*?#7Vs+N!$YG"
        "5d^uQB_ZnrUc_Yra=i4LI-S4r!%(lTZmPn8Sg}|mzxI(2{>@BXt(f3ZTyD2njW_ECWF+W|NbuEHDf|h&w5&Y*=4FL~#w+39C>p|!"
        "1@~1}b9?D>&QL9t7KR&H)QjzHl2w7c5X+BM19Hh9GEqn6EnS8NjKodcAU-9{b|(Tv4M2rxKs1<$vuydA<pL08MFMD-=b`tt=l@#x"
        "ILcz8i}1Whxs)*RXA<9SjPEZ#m04jf2L;oy8odVE3mAi2p-CIAYp^X!E0S!@MQsYDTGoA5ZzP@Nju(Mnt^Ckq0xV*Y<ou$eSc#jm"
        "lnaytja81FdA``N^OxWH;bOxd)H8#WivQMRu(*QXM`cs*8W%lTDyboMiXbGT&WgR%3gBdgtPewzGL{MMq1;{)&}CJXRp7s)DLfbs"
        "b{Z`Z%TcNJ9@;dPf!$_a)f9j6;!Rls0_Fi>o+NBx3P{T9c++V;Y>wbSn%MDp1{N{ia4j-lfjZ$ZwR4I_@ICK#^&IwX%WH;Gd(tA4"
        "$UTs~{S4*@8{=)ugy-Hp&(<5DzPvBcDo6*0*vM>cn)MoXJ!P=9o2GR#n-s4i4TK9okS!~+6gH!8W;zB2NPQ4Rw5L(1q})64e7V$)"
        "+(@HZAR^f42{YA-B2kUP%To@BW3ndK$i77af}HT`24p;tcL)vM0YNZtcpNzDUEi1+k6tmZc>?RwbX~WTEQpU$mfRTTu$6_S+&Elg"
        "#qNFhI|M}Ul>>(Z{{$?EA4PgiJyb^<1I>nlZDE8|59EYiMQdA8%C{i;#!B%cH5n+U-ReKXkG1*1a6vOQcg62A0q$+t4biu>0Mq}&"
        "{v_6=a$J%uDF~~5%|&31gN@WiL)r&2+&vrN{o7J8<YKrksowu!dBc@ve?;Oul1)PGZ8NcDphpYenmg!r{Dc-CDswC;B|<tyZ^MkH"
        "2p~jkO+U&Zth6791$Lx%eX6}8tQf%;mf;Y_;wEhh0e8<xQWa_(hjFN>8JX~kvi9DV1($<b(3uwj-6PRUU?Q2ER7I#g>p{aQ9FOGV"
        "BO(@QTLBLND@$8(QRa6%Xrr2elaPPZbAe$Yps*uFNzgj0CFQW5JTLD7O?_Whw@Z53yQTN40Ni5cZL$Pn2GosD93$x^*qWL8r<^%("
        "1tTeVq#}Jyv)~|(pb5MV40DG;$O9l`<g$6uY%x#afgP+$%JEyztAP*^2Jf$mUw|meGl<$eK%RrImZXR60D5RC&<bX|?QYf8-Wq^`"
        "G0;91Dyw}aYH2M)<;L-goE7VatvPD4zuVUv@D4}mGUOAH98VM53XI`^;`=pCgx_x3CF^IAYwhj{TM&WS5lE?obxch2L7Y%Cu#jLr"
        "aT{@~kD4Zduo@C2Ff|4S$0U-!N|J>GH97^fT(C(FqmLiUx&|$uFmn74AO25wjJQ8kY7cQ#KU8y}2PD$l4<9=q^iohChDlL>wP2I="
        ">MTs?>>a!?k)zVLsA$a=4vMFDRdxrP005&oq=T!7Z8ko=<56dFhtF1d8Q+5CrJ0v#C8WiKRLpz=NtFYb<r(aoSZWXwU3C-Ew;{lm"
        "Jwz&Nm$qI=<EQVDv|DhtXP2?baxC2s^pyqI4#HR+z?Hp+7^~fKjI;uUzZEPCjgd)??*Yx>SpH|X5X&6+q6H`YaDdZTGeb5&jN4<7"
        "GJ?DCu0Q)oAACy}M5yNz|BIS#sqc?6zb3n)Cm{^3eSMtXGm!RcztcFt;@`j)beumyLtt~>GDqZ47HkKZV1oIm!x$@lGGY#go~ltU"
        "f?RYuVp;OWni7T1mNt$fk|eP0v^CD<KfYdiXJ#G|^Ih!)o&BD9@2=FBj}YyFAk!6P#vH#{i&jyCCin?r2>}VYd5{YkFpLZ|y~qVR"
        ";su1GLGSj=xDBqxWliLqoZ;SsWY<+ds|l|!UY?#hrOqjIRhgqnwTpPyMzC;meXjP^tgo~dA@FPPnFUE<?xSQ$oFK9Ez?b=+&+owa"
        "D7lnl!<mxPF4&gRVr+?XN{LI!l229%0$`0+M}Tb@N(uGS+&EvZ3>Y`LaD$6JE(*5`{+PE}i~`4TCyhYP9I>m)m_w{@4FS!?D<y6{"
        "2<byEK*i%a&@$f}0m@cmWRJmXJ5q<T)wPz3n}vYT1%&7Vm~s|ES|hV}S$L8YTQh%Bm2h0D9AQ51c^()t>5ly08dkBQ&I+y)i%3<K"
        "F6=;OBn44PXfLwqkqdFO3cp*-{VG`(kehHIJy}9!VF2(!?dmK>F9dlUsC~YVu!!T^3uoJ`AfN~#8f205;l?@(w1iEbLu$gRsEZs~"
        "lK{g^jIH4mak)du@_+gG`W0neG<_tGKOjxzaYM6i8OAv>rvmom5hO?PEht6A3g8tG2Z4~xKJ`qGw%!^#WE{OjQb(jCnJJCYh0iK0"
        "dbL{l)LGwjB;cE7%L-K2G<z6i<0EzZQl1t?B5hAAIIs@`d>Q9WLJtorx{CNm@Ef|6_S90qza(FG<j_XO?M!^zeAg(~F;#0GO;-(z"
        "yW*+L%K_>dt=D!}`&iba>s7N}?*U0pGHE(|mV)sa12e);2dIq10M;-IXMrT-P<~@15|igAq;xAlw%M}k_QaISxyo{9ijbLSAENl0"
        ">ctN1f?`oPGsrN$q9d;wGNTx_z0MxIdc!^99ExPlQPSk8R9$)kzKIl`Ko*rYg@i)^0ilU$)WQhv@H7yx_VvM9rXxb}%RGrxtcr$9"
        "uD}r`_$X<{K-yK;I7CP>A6ih8J0m!c9V?m@x*ZClT0U9>?EqTJ^wKvs)kJP=jw+J$O@hTgl31#pK=5UUQvNDiqd;~Nv}5tDy%hg@"
        "_RZOM-=3%6y<o56)%lCn>gB7mv(<Mml9!CF&c6MIr*Zn?yH^81#Ax4~N_CuB%sA^%esFOW+C6f4!xjGg=Hq}YGPZi}v|;_~S3B?c"
        "+6Y`}43^vNtWchUjRJzt2d~KMX7?oEYXTgs25qywm=W37H7jt}&#5#}m7yytm~LQLGJvr6sLb9;qw+$I<gFX?oTOi_-IeA|WPnGf"
        "q@#5J!<j)$GD3v4eJ=BMWKF=T2aHw#9NE9+0*oSnw55+WnU8KE>u2V2#m3{QblJ>>`iyfVS5|Iy2v*H>GLc9*H_^g18dP==*7kL&"
        "SkHXVlQI^qW41sJCS6J<jAZ6|VC7UGL?@+B$sPz8;+wjR%N)s8fqIgqsfy7emN1~N`U{>)*AQ6Z!L)$TY_$iS%yWD~J1;go@!r!S"
        "Uo5^~oV}n<Ck|E`c)cgpj&lYG5|am7vc@9x&uSY63_6P0kFGF2$=JFm1?EY15U+f}`uA+7L+g>qyhGBqH);HFRSJP_m5vU?tY>+D"
        "WAd&>e^gNX|029^PYu+8@KRc`-<WygGHfgA3nOx#j^DSyG+{*b{Umx&0PMzzCIu;==@;wYE$2$r8w=-Dn}8viB%PB^Rd21eB-syS"
        "aQk@<(>i;=H5dB;6myH|P`4&LwgA@N<2fKDOnOM*mlcb_2hDI})CoIv5WrRBhxWn#@bKt^hMI$bWdd6qDqdF^YMz!aI?RLzIP;ca"
        "e^#d`y>sw!URwHNu@ekUJ$YF9C-C}-&Z+14p68s#oZ~;oJDI1a)9y$u?f=ppm7u?IMx1K>l5C)}25wRIVzmF#MI~SySzJdXI@EYE"
        "aJEmX`90Y+X=Y8Vq34juKORv?Lb+*V_N@G~bZ^@6NIdGen(`os-y49eKnZm14a{ki*PY)f%eLh~FT-wnXlpQTog)-eVjZ@e<5<+C"
        "rsOeXnh-rO3|q5qf~$6`*5i=L!9l7F%A++T^Rhe7h6M{{9SjXk+*N}$(;mfO$OOlxYL8HN_14y$WeGKA&>50>@^ND}M;a(Q@=<Cq"
        "jRFiIwHr}{lZas-p-JRVx)f_7ZK+x9u_>T;L@=Zbi~lM`m5KALyCM6s^leC5!;y9L>as@H+vjOcNU?fsnScx+qC;=qqM2mCdt-Z4"
        "+oabNw?%nh$fC%bpE`cDCov=y@+*W&DnG}{2rk$+S$=%XCsogQp!G}5Je19YUiPOJnH0DTJ3^`3z4h8NDfif<hXr-E%IS3kmQ_6$"
        "+nyE;s0?&pk*RS*!c{MnZhCc+K4@@-D4`Mff<s+;B?Oe@%#1KVk=3KraCpYVR`uR(gl{jrJCd&38GND2Frio7inx|rYX4CNQ04?3"
        "i?DrIhvGDVYR+A~ndGUJT7w0un?RdE95=}JL))YpW=nPs3oy*Nw9s|gn$)f_;c!%@>%)>DE@HL=Ez~yM=m<3#KHU$4m>l9TCC8m0"
        "0ZPdW9Hq;0-C#!#O%IkV0V$`%(laAg4NdKs`+R%o6-QcX>^X*`COT|7+Fk88c(iZS0plv$)qaRB#dj5q6|tRGFYzo~Ix_VL_P}+5"
        "uFYmw9FuX^(h^G*jky*n+Yl*uYR~D+2F3mp*J@=;LL?x4D-soGtD93KtEGXdpzhy}X?o4YT~?I^=DfbW_F<mZ8NHDRC{Ta9;6%sK"
        "{lv6Y#<56yvdz3l%{2s`8!RB-SV#9(0I5n;v%v^G`6uF-4mZ0;*Jeb1Gl780*cS)0hsQqkyQbQeLe@#mw?#<9NzbaNi;L*sSOyXF"
        "%y%0Q)k@JRP^8|B>)sZ@KZkwMll)K@OWS_vC?DkTKYV|IL1DwC7vlSwx^(*@z8`ucoe-#(29L=JvO@RB8*duw@|8wIR&NXg?GQ!C"
        ")Y<~BqW>-&eyAvvBaBuef|^+)MB0`KMFd=kTI#wKYH~NxBAWN{^`A&qasq#0zuj@X40M4aps%x(-bU0f+z^^}``tA*BC5MqV<i(n"
        "7(lKHbR4>VHe{8-Orwily<j4{9(E<jlzN~FZN>wqGAL#n!_2^>B?}Bc>`XU^6!^f7Sx!<db!66o_3fv6W*nva&da0v&fD<CS#p`~"
        "NI#<CoEp3Cr&54Fpmj!l8GVD95#R^!&9G5;N8;uNa<!PE_zZ8`3fXFe&cW2#snh4vAXWcx2L+CL??`tqR<O+yTmRk&ke2sjTY@!Q"
        "UDhBJT8&I-eeZ~dVl#Vwa*L$Z`8ddU$J!jFW5g;g*2J*5PH#VrjHZpbkTNqe1SKglnkUQMD6&@46~Wz&<C`q?2X`qe8sn&z%NM(4"
        "M?&|ax80C%h*Po^SC%RVl|?!hw{ALGWeA6AU!Q3y*RK`_gC5jLT^ZyXrrNsl3WuU9)>wBBs|g9q(xu{-PA-p7RHawTNu_^|Qa(>1"
        "ds+PfMRXmLTM9^9WA<CSj>3D&r$D-UE(n>wwu5DLwLk80u%<sbW_4BhMvfU=6`ez;3DeQm#CZa74tUcA9Jab@Juy<2mbgJL5{p4-"
        "aHP-K8U$utah~iUn*fi9E*85)QLIOfk981@r%uR{yCqh8yZ7WPK`njyM>ZCJtJV$1bnod&kZ_rvV+tBYNf}3xjtrX>o=(v)?Cb|t"
        "t>l$JsCr4Zg%#B5E{}5)^9@_s#!4CKir^)1;M3cdbJgYA8%)EPA~5jh^H#IxiUnT1-|?mXNr2&`Du8m#s-F1$frA;91)kw?e-?AG"
        "l)uPe_n3#1(+{v>gm*OYLz>#Mn1SK|%sQ)GJOfE#pwM&`_)-zX3BIUl)ZPxDe(t6W5J^Gk-pB<^jI?Cw(+g!0IHrReMpc@*dX4m1"
        "7^rD|E}Pb;xFtu36Qh%iK$XQfFpK2%^O>bQ(OjP4DKwNpX)3=t{r=knNXCI*oF5Wlc0U1rc6xex00AVEb7b8FmRAaeQ8gb`<asy9"
        "AAlWJ>p*vDU>}%1d8fwB5!IeRew=nsAb7GyP2f6K&L&ViNz~dVq=~KSHTBwX&t&H~{FAvkj_O!uj$?QdFJ<T0kdLySXbu``moBLu"
        "q;(WthbRs&GXg_}>bY!p9<`fdkEJAC+7Bl&N@I?yIB!etlQ!LrMi*S?Sa5v?ysF;oMWuPkOjCI}34E-~=pfOt-};en4gCYVK8m8N"
        "@)_1~_`7FM!l`nfxlw}q0|XXR$PPiP;@vZ$pKMA-F+Ir+9AxZ-3qAY({16sMTp?h(m#Y&_G%wE%VwqUUA<$!01hcg+2cbWRhl-_l"
        "YE1t}G;|88QWNTj?ByGFt<(L>FT;%3Dv<j>$olBtu|7J#JpTdz4UXyb8jdI>${+eEu@p|nezHjWpRc2j@BZuZtvoz+wS$HKl=Xb`"
        "1OEFLuZQQZ-PC!UwUID%o2er#S#q&{r)B2)jgpz86)yDt-8(z#S4G|%XQ-<tS!24pZA9&@xJM~McE<-lf;9M9qazpG52A*x1QV}n"
        "9y@~(QIFtv^^hOJE`t~Q=d=1ffF_PXpTzKZP(Gjdp#bMk11K}XAF?_)pLwa@RZ!JWnWikud2-=>CeX)U+aJSFKeu-(%Pp|Vh$?4M"
        "L@SIUEWSjMUZj+itG%fC_7Z2K+6+cM{|it{0|XQR000O8001EXwfc{5-UR>v*$@B#8~^|Sb8=%ZVrgb?V_|G(Wo>O?FLPsIW@c|}"
        "WG--dl~-+V+cpsX9w7gL(7q&qU%GB<hZX^rG;WbLNweAAFbso1lP8-CC2AyP$1U>TcSlN=;@FAi0*y#feD3wRyCg}jc?OaRlRIxq"
        ";g!kROdE0Q7|$hxyFx3WJX5)cx#jV_DRarK0yjvK<nZ9|AhjlAl6x*RcMcp=Sz)Yiw&>8|!7rQj5q|L%@{@@z;rQ?%?6I>_X?aHA"
        "Zw?O_{*-*fP^`Fcp8L|ZH%w^-v^I*;7RJ>(^zZD;#pU_*?DF%wPh)n%j^eonr`n)j^6tg=NslEcxrPi-T}ewcOseLCVoi&&y6a~~"
        "f@ZD&p;9HHLC8{K0phkWXeU-!vxJ=I0+M+DaqlVM^L@bHa^YQt{TrZe${V}xp;tH^?y(v7ViDjA3sOb*ZNA)(j+ifZV3K==!f4GA"
        "r^3Cl;LR=M>MMqc2abB$erl{x`8?u-OP*8khkanx)xnH4pvrTEnsEoiiwUd2Z45|6&bapo-qE%{(w&9o0v1L~up4MDdExQe3U*?-"
        "%1I}B>A<u4eSk%WTFnc_OR0QE<e!pWYyCik)_#=5`5;DXe_{dp=x`K)rGTMtF|D{(b25%M%vd#7IoHuZ%%mv_?dmI@lkL?$hqx8E"
        "O!-A+l}jTKV*JObA9h(m4)054;bE*j5o%X2UeirE`gjRb^jx?&q>&rO-3shFESN=uo2<VAHD7p^ft{nqE&h2X+O8;_T5Ag_SfFHV"
        "?^gKn_9;dbnxT%t*0GNvp`A&6#_xLU!>^|y9JpI&8F;G%CHp_o7*E(9h1!*Gl@O54KuV0gIu^&9+gHY7ZgYzyO@ZJRNV^#NyCFiB"
        "wbb1JZ1(C{M<-CkXq5va!p>%T9Tb~!al7JHK5OG$X;q<N8hM40`XH;%s&uZs%8P=M4Xta~2gowcG+$%3b;_+P%B_9y;ZKG7zSO$9"
        "9@WPew(>ZW5S9<(svGpH{m_Egal@NNR`t|@PKUx8EODQH4`#L-GH2nNPSBH`b;4J?H8isjhfAoV=qBW<$<ej0EWj?6qe>s-z|XD0"
        "W8H9H?ofGy8y=u37baC2ac)NMC(IOHWklXV>-rO>jVU@p|L_!a5y0@N0zVbLM7qi#Bg+ybRx)(_8n?JZ1|_p~3$aOza8ke^%TD)+"
        "AeHlq3T6}PpQ~>IxUFj@DoveHC-93-6sGcOcQ^-h0N=%56OPgD`zaO^Hpl<unnlWu34j{BMFFW<LwQ3evI272?^;P-Ka{B*oZD5d"
        "iTZ6Z$JpBumxj{ckBuF3IWOG8cxpninT&<_-xA3XsXB+J==S`@&->dOT;BPb30wBjaHpu+RVUa=VI5jM*W8RKM4$B)EV(X&hpLh{"
        "E!KYV*k2eseU;Zfx(R2P6UKsli>!gt&G7C;#_cVeV>B|nSIO9g{x8Q(w%U_@j@tx!<GiU$A9X8Np%IjQVHVW)xA<+0G@x>!OBtH0"
        "NV}1hsKuia)Uj(h-O7sHx!F3vF)#d($mUU67^A7II+7{WJsN33Sy+NLNS6KZab&7a(v2cJ?<0&>0ALMAotwTMoqCR!Aw{6sxU#G{"
        "+lehk#LpdE*%G0d7Tlv$#9lANSbL=SIR1QniFM32b^WRI8%JAgnV^}FT8ym~cYViGXb(`1F9H3=V;h3=VVvNmXMavF&T)?W7ADjI"
        "jfdfhrh@7TlczoO-N!jF&X@0F8+~m&8eOhO)*5?+tBb~9VehyaKXJG{%|nNfN2m6fG@^_r>nRw=u^c&yvy{&)4TSj0l|rD0(^4~A"
        "=Jk+FS4v3PbV?o4bb3U6K`+WZtwvo$0MC^U3)D18YGLRXRXkyBbNp3*|95kaE*{kTc)otzI#DYV>Enc@3GTRIwBLs>7w4lvCcpbu"
        "e?8D<1@`FpK55-Ujm6RNV53v>ihgOb`TG7%?Y3%6zh}w?)732)$N~En)xT%S_C?o3$<9OJ@S?}!w}rxh`43P_0|XQR000O8001EX"
        "+fWH2jsO4vy#N3JB>(^bb8=%ZVrgb?V_|G(Wo>O?FLZKYX>Mt5XD?r0X>MtBUtcb8d2P)x4#F@Dh2gzY?_hbS7eL(*7k~|+$aRxq"
        "NaD&#k+?kpiH+?IpZ<r42QklecRHRya`hI7ec40UVuWfWS@HtrQ(FZHX5}sU2IntdjH+mcDJnk(MC`l13#JJs3Ca`~pArqf(4f=3"
        "y~)!TkhH2*SgG>Ov1uu;bxqgTr{NsM=p2l2!9N7AefI`XO9KQH000080000X08aP(@RkMu06!D}04M+e0CRF<FJftCZ)0I>XJu_|"
        "VJ~!YVQFq@Zf7rJb#h^JWn^D-W^^uad5sv`Zre8WJwX0}Qy(e^uDc8y2DtOnwgP?F1=z3`*AQrliit|3KvGWJApgE|c#%j+&L*HW"
        "d2T#BcP7iS->4~1tRi*aH19eo8)D=aCiqusxD<qnij=bL8fMr#*2(fj{`&YgsilH{y&@guO20Rck{qdK4HryrvMjs3xxL|SCzT;B"
        "HK*H~TFDlO&8gu>;xT_(jK>{$Elb)wu(qX}iW<6+b&c{0U7NtsNLm$Tuc%>Ffy%ClHuTuAqUPd(sZMcWt_n;p7E3%c#cK{CJq1%s"
        "GK`{Jz%fgT)!Y;j_*M2!HoUw9&Qgk+KlF+f4qmjfVsNjD)wFL+5!i}O$(lDzzbadSi!?V<U1IIRwTcL>SSeM7=D)pxvq%Rx3R(o~"
        "29de$Sh*%>?4shph1}~wq$)}Z@w3|A-Vpe}^*>9Y`<4;fGz2vL$^`lJizZwUIFG3hL`b43L=*^L6UE+XSQ<bQ34W*>`A8}yyAH2o"
        "BMIdy`KX-|WH*n3Jk@UHcpBbvqk$f=`wZ`{Om^63R(FjnN0J>JLj@)Q4bk>x(Ne){rcF3+9wXK=n}G?VIiH%vtHOjQf<-JjcWA5a"
        "YljT{uN{IPuSv`>B3Ppt$$yuEtqK0`dHj#;P1jpnJ-;{zCb)IXnsCEf>pyhD#_saq?}&MC$qyi7e;AP{2l$ZPaz=dkLG&B;gr)$Z"
        "%>y%e=0VvCLn2Sft;Fojn>XZ>&_?wort)X;49xqtd<4BcynF^Y5`Ki7*%C*A^~a*z_Q*gx27oL*WtDD&xhv4}dL}Js`h0f>=d8(a"
        "rER$H%X&_2^K1?duAgH<<Xz<{Y<TZnknb}O%?{UPO*pW)7ao-z=3G~!ZBPWGE=UUvVwAXk)$VCrZD`jqQ4PUgrBT1f=-OZiP*Jji"
        "S2pqxXe&=5>Oj63ivs!%$gvza>?A^=48k4$p-s<zRZ6Nn`{;^VDNk5idBud`=0eI7g%(tis)r`@kaW>a=9P_7|9Lzn1683*LU?4M"
        "s$n;_r47q?alX$YQ)Lz&H9TN+w~1k~aWnSRL1k%~rL`^-v}G}cuY_Cgtr!)Is)uu8tHqRKVWL_yEADv+yKPeZAS*%`Y?FHXGP|iS"
        "+}lYVUply&m}Nd3Ow#muDGwnEwq!rY;FmweLAEAOA^6c6xDA;E{j}y!0cxm%;)c7`vRsVTr=d#DX%@0?sYLrg4^=v4dKtGXZ@rf^"
        "E+-)bVY*<!K<uR}W_mTISY`(YaL)*-*ctbMZ=5-BY6=k#aqfNN1|YY#!q%ZZpVx7_TBo}E?VbO%v#Q?R`|T0O+(JW)3?N#TecLxS"
        "LwDZ~(w+)!Sf_WkS9)1z+7t|$^JVwLx8Hi8re`(-Du%O9ffHIR0A{xbKqW#oW7Y2dN7zNf(;tEn)r<Fj2*er6U4$GjVI=$jrG}MX"
        "Itk7o=kJK_JIn}8Aj@HbgjvdIF-}iVAH9^H$UU~z(Ha2~8zaDkG|mBG10IN~<GeC{5@0AkdoUdP3W9$qf?Y#}0FU<_OyvxR4zJvu"
        "eJgBA#+$rhBDZI+$a|7U`raj~hS|p*fwSI3I9z>jDdF1AKCUed;K4>-2;BaE-BNZ2Y9&-V;tq4AbGT+rTi#sKAurUtLW<@RO&zEu"
        "dOQ@9py_me%k-sfnNhsVMjl(w4ToU#l_(tY2t`^u&Ww(n>FDQQBfIMlj(oLuckwl7`Nfj^kae(N1AM8tM+m(8HlP5-msoh0K=NpC"
        "B9;Sdh45NY<te~0l)%$R-e8YS+a?7%D0C&%ABN3S0O6S>FN3A_nKZ-5gwvL2Q3aw>IwRq!QJQ>}b|6s3b{v)4^`<EN^@<{;R8(l}"
        "_<#ilPn4(dgyTrdFGq=k6KajrwzH<ieF&`+uV5sx$uj8eaU18lO+>Q3BIOK^%j2s;MvAG75Qj^ec-(>}sgR|Nz!H>Z#gG|MUBxdM"
        "B3$r-25u3fFc>Ez!<AO}AM@A9HA5aI&iBb>)5H!nxJca{iXEWGO-vG(Kp={@F&VZPj7j)O`#m;@2FvB-?-o(1in}m=aVN%a?8f-n"
        "cd2X3ap)%58^Ome>uPR<@_pO=MDVDAFWMyO=2r|dvHosm`>4^6%LpI!{BarK<Mtn|BZ^@X2#Z)cP6Uy&teG)FLv-yXv(75^CWM~v"
        "efIHS4H(*>9pxsiG35=jLov9<@=ThfpY<Hm?ajYXO9KQH000080000X0EMl(<sJb50ObJy04V?f0CRF<FJftCZ)0I>XJu_|VJ~!Y"
        "VQFq@Zf7rKX>)XGY;0k4X>V>WaCvQ!u}%Xq42JiU?j5X5tpryFrbA+ajttNpr8(ysNaDkBB>MLFdIc4!nQX`L|M`zGW_Ner15FYr"
        "+X&GLGT`@w91>=EfJ_1*O~ir&5Ch{^4XpA02!jSr;BC9>j4^G~HcKZ6u{+GUXG*XSo}7&%rewR~toIa$^HA8LmCE=+@``piJ~I=~"
        "FWY87uL?y%R4T=Lk5tO4NF5YRlQBr?7pZQCxe(ugpmhi2tG8>oef(GG+XnO~`nJvt{k(V#E(q8d#ylIaNw$2JjbKtsPNZP-@T&uk"
        "(o{uj&QZ-0^0+uL&t)45UfplCG@QiMe{%!o34O>q2R^E9?1LJ7x%<VJz#cs*hWZzAE~w}qPskn_v%~HTv2L3mP)h>@6aWAK2mk;8"
        "ApmS_gL4=T006Tu001cf0047xV=rQ9W^ZF*Y-eR{ZDB8Ta$#w1X>MmPZDnn5a(Q2Ha&KZ~E^v9Z8eMPPHu60{{sXH%oaC$(+s&m1"
        "w_Ctzo5KN38#KAzgJB3<iM!@mD=8vndt2nc-^`E{sgGUTy*mMkSK`cYzGpa66h$AlyewJWXt|ZYb1m!Y<!`@UykkvUuf&2?wPyNS"
        ";BC?6xXeDh{m697WhL75`LpNG-qnRr=d7)}s>lFKR;+GWeJk3MH`%7;&9zF|M<7yitwqaPp#fVl+;|JfrTPowK>L=qfBuXa9JUhc"
        "x)p52^P4-~77MoOG%ISMh^xHha?A3%%Dc7|m1f`n^M?y|Bic%oOx%mSLu$Zb6h+9U+%|O!!pN-%Z|gR{9^R!@#W?IyO&+J~t|A@q"
        "5+U9qo#)Sh5vACB)Ynh-uNOaoVt5+^G$^JkMRE=j!BvPg%Q9I>on^5S<r?JSN}R(+sli#WmRC3DOoA{=cKX_3;-$Jqv~iO56cWEC"
        "*M!KvU9yt^1urm@66`l#cH+mjt=l+477h5eQyO&4np#OvzDT2hCNh{Jf4~X;nOR9&@u`!oP-x>Ki{J86i7^S<X-ON7R#eVQS#4nP"
        "EIncDOK^wgn&T}ynu9OvmS-2R4*Gh}TfP<GOYu-#V<@q0^e#pcM#ZVDvbEs416~(khg+G8B`w?uxw+P<c?3&(*NCMdvMy_`zd8%#"
        "8H+{?y?G2nzl*k3G3{7)I^f0d_`l!?ctFQUtGb1NHV$T0wqQyZ%aim3QN1rhY++PcSG+|RQK82{TxvAO+q5(u-TyGKhMzcDgytFC"
        "y$Se(nD5%kJC7xjuoq^dRM(C!$((=_gl=L&4f%$MU&{^XFQXKdQ7Qt6m{Q4t9WB74OPTM^LGBg=h>)3R5?t)~J66bCTah7%3>qaD"
        "$YFmtylCB(W%M~v%`~b$S|%I~7h6aIvmuh`<BJ#_M~_8ZF7196mKVIyq5zgJuf~inoooSXU{zLuD=u&s$UZ1ZDU~+bNpY#j%#w<e"
        "WRxy|wp5UsHLr3J8?c2>PzmEzLFCg`HKo*Xl!Agzlgs1NE7IAJ9A42Uw9~(%#mv&p{-xXfOZDv1=kqCO!?7@AMjudnS_u^dGun&@"
        "Q)~rg3UI9EHbQ5rTN$0i0p%nhy_l2%r$cnUWM&g<-;_6vsEU{fdZ;hih1zW)v9vOW{3tio7Au{E>Ll(U)M_TR@wHZL!2?ZQ<}0wf"
        "uItnjLX8PizL#th3I_fysvLf`nFj)m{9IRJSYrau@Q|T*j4a@R#=ics?Hp~-FnD<5vM9N+zXQRH5&B2qyA%lSez=2=6l!1G76&8v"
        "=Q`lU@5Al{v>=d)KoT&M3rm~M0)LN8P)6rAIs^BN(D<-GwQweU8FWB`6docO&uCq+0>1T?5U6K4^DG0T>b9}hP1S8h2@0DiwZ~58"
        "X|8sfHrAq2b!%?p3DReun}>jOqbSir*0tJKLZA;qaJd#n*BbgM=v~Y$a%oVBMMLB~jEVQ&IuM4etK_FnKnDp$R#CGmVU0u4{D##l"
        "bU`Ky2pg$)5JM0%g($j4xt<qP0g(kINu@(~FRTQKs|d*Y$j9+f%8MfQ(4!zX(!E$(NqS=(xd-4{7}S{pR^p+15w6#6&#+Z+e?Qi-"
        "Ll3RtxuP;VtQcMu*u3{-hd>bfS_}YEdX$>F9d@J$PESc~dw_;1h1M6gVftuPX<!qtflbq8Td}R%U2Hm}nJR85m}Li*VoSlkfPl-p"
        "Vo)!ws*uE}b%WDModf5EQQ&eEp9Wcp_Vy6Ox)oyoD%p4cawGUn_6Rlz?l3P_mB`Mb4>+SK)JO_S&9K7BYF#^}4SZ>&pfOiNQeYp?"
        "w<EU*l|JBgitGSpa088!G=3%2CjAc9W>p#P%kQ&CFg~a0c`AD(2z(jJ<64%!e}&3bca15^3J)$izs57d5%%w*QOh&x>BKS`9<8BP"
        "mr#9)1Y|w2JQc^U5+4tkybMunL`zZGEfS0eytO+ejA9&iV&AgUgSGNQ*EHphfr}Y7K`*V@gBnDxkt1kI5Fnf^5c$4_0&j@p`|V(Y"
        "fNl3k%)<>h4k7xk`|v2j?cbL|qj`lL$Yo=y<bjrB)M^mCk=@~_tMLZ1AnH&g8dHX8M?{z~(Qi-p!%hZVGae%(&Z>jmdgEf99j+t~"
        ">aoQ363!g5Dg<_=*m722OFUN3HAz^^y;_aKW0NiJ?Xg2$o?HzSnb`c?OUJ8S+}MZ6y8Shy+e7v_EZRjsZt{*I6txE6snj?qAZ7RN"
        "<?-bZO&F#djC)#5FABc>S4;+hXl$2*fLtKUg6G#uGa)(b0A_<L-wTc84@5KyAcMp{rY!qZ@s=|cFUVJFSLN4jU18@jNctl<OlCTx"
        "<Y&_x)^s(*vD3QcOp&&opXRMO%Hj+zg`Ba@a)?x%nplt$Iyj+%!>+1$qpoW`c-G~)Q22!O>CO?m(J2^n4XO>g?v02kv5ikC2U?$*"
        ")1+^EaGpcakKvW+I>YE#2B#W#xfF+X7^Snh4LaR)MpQDb$;^rO0$UZ3)8FMxsZe-M)P|9(DSbOu1@dO66xD5%K=-5p!@~@=jZ(sa"
        "11L5aglisDmik!5?%I=Spsy$>wJB3lxEXZb0&qc;2SZ7B^<LqQ&GUQy5uw%bI0S#7pnwBz1l&fDM>9MA@<sIMXzIA1r-WzNnQCE="
        "5;%BwtC>?F%8hEA2evtXRo7)~Vo|C!&iDk`U^2UzYWr7+Er_)KfLj^#{d&cV7Tw4_J^&<o?Gs8NG^DsVNnwX=JU8k<b0g@&Drw=I"
        "Z3v2o;Th5XR1oY@k#+Ya3&U)$6~bU~j*nBeJ;n*ZpKPOdQo)I6^RjxJt%9wdXtEIVVawr#)-zbs9BK>SAeEwR#<eW~dDu{xUnSIL"
        "N?1*bx1;9Bv}3doMALlg!c|Yvnh*P_ITLP@6qgcPs%=~w+`|d(A#?U*W^L-)JV>*|)+XgEXlqg&=OpYqcJkjcge{P7;JiL?jDkgN"
        "maFi%FPb_3Lf`0#jRqv+RJ}JT`{@P&>`BPU)cgK#75ov+6kd;QsAvu{Y9jF(&0`)AYYO;;xr{@3a5iJbf7E;iuOR54S?o$ouA`R$"
        "XldQyI5B#E@!`XGejiaTHVq-b1n1N_K!CtM<0b0ypi!l%sVJ?NsWDiuC}d{Z#}C`}38UUo?SPG5Skc(E#s)Y312S8z-FqyWXZE^Q"
        "pC|H!jt>z1)b+r)Z$AkPnIW0sOm@r;P0Z3+H=pd{)A?B9X{eD7notK%ILD3Jt^<tB3fVmQ+&JFgff0D~q%-`4x)OO^6-0G<5Msyf"
        "pfpA^6&z!OevFsR`3siKkMS~JYx4JLPMxslQa0Mx{T>4avw_UKvqo@Cpkqk?g+_*Mm6<OzR7T%#hyt)E&H}W+{Qj(mq$gurlWXi*"
        "87x+o<5S?{io>5c+KrQKj%WVSc-tIoNP^LZU$XCkS>UR^tCaIloSkj?4y3~InZSv}mXY1qFTcF|2FUgLfPQDqSI|G<pfVlOK0GvV"
        "mx+Rd;<`yCR5f5l%N#NbXk2CBpBtPjwPG#rGf34NeGR37p5lxNKWQ+p6m@+AfwO=RDGHp2-~h@p6GhQm3H=E7^lg+Kn{6yXChmEz"
        "%N@us;0O)<OIM-TpeBwV`tWU2LSsu``B=RN`3UpRkkQ07i?Rw@S&Y~a3TL2$RU$MLYcC~yc8Z_l4B_#~=~t&GU!P5&z3zsa_K)ce"
        "`xk(J@_K?CDA!Jz56;141+`oj*~F|X9GmiX=bAuq4|$MHq<8_c9gnniOa<Gs>&H=s$hU(plcqdPdFhG`d1Rd_{6Y;JGqaY*_tX6d"
        "mm~)sW8F3MUf%l|J7RHcwtrzKq`~4bJ5AWjmy?AgxjZMNtHJw^_$aW&IT!=_Q;W7hqXgjsTw1k2WJ&W*U*V9xf3$F4*B#NX4N6RJ"
        "Y{sv%z^Q+M)21D*!^wau#y3zQ>V}obP|K{<q%d_yUvupcD0Ohi6te7E;X?#paoo`)E_{LgMy~6emwy-AEkFM8><#`4o1YQDt2g-X"
        "0KnNp0I#N7n3Y$A{Ev_GORvJE&w~LusoD+1ar8|Ol71R)Ze84X0E|qllt+b5V9QM0Ljjlg3RgL|#OI(~`tBwGO$NLY5O9L0J-?@M"
        "N7)j~XKtO6&uy4*oD8pDA@8^N-&bc}zy6zP?@d=C!w%iN{Fkk1OJwnsOc9Ff-7R>JgFN!flPfr*uJ8}CeR*nbPt7g%l`G<cFW`rr"
        "5SVDd|1VF^uO1lShiu>kKa*A7Vx{cA%Bb(uIjQ{;gXk(48oTeJaAUIhs87It=6=Cq`z_3ZbR~++=Dq~+1irh+i9WGCK6hJ4%tg~`"
        "Yso?)8q$Rae?2`Tm%vYwsPj{IQeVftzekDrK4geE>X3)G{Sl1POl-b?>Lqg`jc`DB`^Tv>A9mptdFO{dj`rCHXumOnZcsydwfP;8"
        "99w(vhEjX9;q!gi&^UCwoMJzKX7q<&?9h*WU?u?v5dlR?XZVft7QfTMW*Z%wT#Ksa`OItzFZ_U)cDi(jWUBS%oIH<0E^&D^!`k@p"
        "G&$6}iW3ga=02XYU@>|IwjWbU`<b4hw}Mc;8PwFU_^7;YXbj{#0qXl)?&;r>+ZRx%X)`hS1QwQhH>efJGMW$mgpWbc%)Ron*c=R&"
        "VNso)Ug$M}``VP+4wW@H!iVg`@8!1J+BpSolVE-eO3V@61X&+1dO2Q}h;Zqg>?%;-(y5P95<t=WE2QqL-G2a3O9KQH000080000X"
        "02*Vwn^*w=0F(j%04D$d0CRF<FJftCZ)0I>XJu_|VJ~!YVQFq@Zf7rVVrpe$bZK^Fb1rasZBemqgD?=?k@$zJC%XcRq#IQ_G)kt9"
        "QI-frE|_4+oRMv$@b`t7q(QAW-+TA&-aSIdAzS4jw7Q_IpcB4qQiFPc?T*9NXf3?4;2KqX2L?;|4%#>eZSOpw@c{yH5ehjWB#z=p"
        "wVkmZ#>TQ){x~A9dn~;&C^QHM`>7$0Dp`l&=hBYyXNlgLZlG}l<J0D%yvt{egG6U+0$O}a{iiw_F;<=xR`QyILJu2AS6LQE4APGU"
        "IS=Zpo(FKOR0}T~c!SY%kbl(HQ_c6CmUL#jMs)I&uM0NeGZdO`5S#U<XN7VSjzafxXRWc6?5>?G{dn`B2B**DD>)V*uo~mAQdGku"
        "mY{m06r8Zf+SC2NjnQ<TJMHo8pK*Z$Ws60m?=9X9yS}9iULgyF8Qk;#Lii1W*V+hA8MpFgl%}f$7Dq`}1-l3I#A3^MHqW0hWc2m}"
        "P)h>@6aWAK2mk;8ApkAoAMLOZ005Ui001Ze0047xV=rQ9W^ZF*Y-eR{ZDB8Ta$#w1X>MmPZ)Rp}X>Mg-b7pidaCx;`dvDyh5&u6x"
        "z5^?uILTSdkKSDiH(MZ09JIj2ckty}6x)TsmAJdA^{O7W_kw))&J0P3l(_44auskFi;v-O_?suSp6C6w?21Gr?1%F94g2~1$Fq1V"
        ";=8I$i<WIgR*AY{>#}C;R<NA6(3`LiWyV*mC|j{A%ey9g`Qqh^Hwmv=QJ<DYwqpa0hViE9azXQ}v?ySDRg1RfDgN1%T^$P;R13!M"
        "c?xth0rYN4cKgGO<z*r=R&}c^jd`1v#f-(!#KfbBJA7xznaHcM<{9H{D++3jIdA}EJkLYkQXJ62)N0qiY<aWI(pCTdpQbGOmzozz"
        "nZJCouFITNyv31B_5B2|s;BGHL^tE#-x`axWgY8r4POgjttmm4svDc4vJ%B~NJvwMf%hU>r<p+7G<21&%a~^yk>@;2c+11M%rXwM"
        "^qgB!@w9%cT3ypf{%NrZs}e-J(-ZxdFBm+lF3plikBK&Q*;UO<_LWte=IO7Zmd#Pli*zlTR&{q3a3ms0Hp&FAi9`*JNFl&oynK;}"
        "HN!ncNxFf#K8iTFk`!bUo9hz_n=QZm?t4Fw-4AKIr8AI|LX@HJ)hjOmq_L*01<w^Fd=R?vwkz&fTCfxV>SsJ(CHzVa4dH)@e{p{K"
        "EjwZOPcUOE&kGC^15=^golgW6KY-<~F7$w{cvM?5b{bb52vZ3Eo@brVq|-G^o3v<JUc{<vCQ%c9Bp{ePs8wiGn1~ke?FS(c)?#r|"
        "OVYS?Q2BHxc2@&eAeB5IK|%Wrg_NVP<XZUF_`&A`t7K{8T4mu_z@oi8B!{lbgo7xx`%h7}kA|9t(;$Uv)@?KxDFu77He;BpcVMp-"
        ">hHQM0Jyp*E^|O)x=ORO-Kn+{-PbP>0h3@p=bovCkv5^k&;5mm8lpB`mnA9zAzdvP09czcyBGS<aZH%sK^G`P4YseDf2>h_;t|dy"
        "D(~o34PK`O*vPpge8$$^z})F4+TP-uO;)aW7B#J?zFA(Kf1iBz`a`1mAyEcEm`B6YMzHsw&ezm=UDsvpuf4Z}BayTI3@5Vg;H)>S"
        "ychLD4TpBcJ`v&1UeF7ABL+5oFNCeNDbG$l_DWh=KtAZ1IbI1tHNga)gEa+fJ7=E;Mhj@YV%120z<Pqy*mPyrXad@B;Di~g5Lbka"
        "5F3(cpl?n95mj(N|LsyZB`}UWk$p;adOPk9(`RUee7WN@=5gslo6v9uUeIr9mz6xRcjWaj^$dmX8$&wQ2+e>W1T$k^Z`)~vcV*#Q"
        "jzCKk)x`!nrB&L97T@I*-y&D-+HF~=(b9S1aL35FOSnZ98*I`M&oVzic^V_qChFq@=oA@1Zn2zMB9_BZ(7uw^jP=5m5l1bT2&-1("
        "4Niii6CDXwsw+x$titAQ41LiuTv&}uYwxC|K++D&F`N$WG(x~NNE*DeNDSx%_SQjI6Gqv5&JLNIOV3bZErx|Ra`26}1kvpC`UcjB"
        "H#JAec7V1XA7Ys01tIBv)QZQ}$CfbZ@~ZJQ6X1j(CN0|eC3s^|K*$GBoqJuoKK)aFZoIw|wM)Uxz(O;CJr3nY&Gy}TH*-K{Hra@E"
        "2*Y-7uiw1;*Asgs{*T!yoUJr+z4Z^JwP(LmW)gNIt@FMS6Knks^oi0?n|8Y8HL5KWP0T9@&%yCP5K)U5RBIPnCK}d43Hr3K?Qqek"
        ">1OS9fis}-1&-H%^<kfjz-hpaG%)%9<A!Zk#yF(2MWfUBzH(kR4lvS`Fqj;4FF*1gWaPgBm4i(Hc_^R-1O-hW0K<!&ZzzPJGm3dq"
        "8Jr~vEbx74X^ti*$xK6nJs?XV2`caeGuRfR_#FBmNyJDkO(@S}Yq0eq2?uSL8{8x3v3Yq~)F7OG5<uS92A_63!ONko*@=R=QDuhn"
        "w4ucLiji@|1Xw0rI#^)$mlzsRntXpDGaO2nYjK}KSkh<Op*Jix-N$0&V3xEMv+39q5|5Dfr#TpqGt{cHw!9Ms>f~9koj|x)r<;Cr"
        "p<%7Q&J#^eX5x_OjGpU}MhV@){SUjO$$-%W_D;U|Or70i-*(WcLd|2NNJsE!DDlMiORdzWK8_ZaM_YS6abTX9ZNTQQ*C0S|5_`%J"
        "3txKObJ(MtS5a2Vip|`ExD?tRM7r6wO@#F!GcEaQykNYSs8a?G6|Y7d<1H7S2?cLyfo(64Ou~mDXG6;kKtpq%gyPS*@)(8sR2Wrr"
        ";;tqKFX7mql-iSR-0{goAvdIC0aBnnVY{Mb^w@I|#58x-FAG@yHLWZW&Gu}Zk{rlD%tQY&F$^e;%MMG6BbX8bO$dy#9Chxb&9oKJ"
        "EEulBc4~%zUBAnIZ#mXO8Yg9!qPOJ5M)+p%5d`Se0I}?xy`JlJUQ1ca*#|ZB?*;imLoGo1=q+nyh$O|(jt?i9hixRRu44@Zj-FaN"
        "@=tyJ(#L2|+X8n6I|InV!9A4q-P|Jsk$_{1g)|LRw?l_5YTFB7@FLs8JFx>RNkYd~!xTQptSj!Y%r(J|#<^jvhgEbHY_u3=s8o+q"
        "Tw)g6fI14FVkh~&g+N-dE(cd3wT+26xwncaG|h-6$jN**Hy|SiGSNjzUSz`abN7f(6=LN)GPfzuR_-$x1@7b?BBS)+GRc9JvRjpf"
        "L4ErYgdVO+3ym%<^h1pT#BxxJzPqnFENzIPMmh$`(;A8o60Q+hor#1FLhkJ^@7^f1OQWT5*I2+amTD%I5Ep-=bPI+KnU=tMCr(CJ"
        "nUxKsnv`BaR7X|>4D!j0tq?;pS5o}xef3f2T!kyziW<3Eh>Q6B8MZvb(lrE$IO{w-pRse*SDQiSk@$oS)#PHNZ3Sw0PMZC|WZnf="
        ">IFXRTM$3!DJ<jyS8AbG2EAnvwxzEGG9_JhZPm3}GCs|74C75BfUq1L;u&DlgC60gEW~s?AQ3wuz{z|mp0t4k8Z^>3Xaug!4hXyj"
        "bV1CnFnELEIa0t&yyO(fFr;C@PCpWubA}1I+*6qGmB<=YgVT%i^B{!W#19-(>Fn97xw+S&zKSLVg4+|hf@?NG=GELBK$U~cgpC11"
        "U!!wSa+I7TbkME+O^czaAQi&PC78(?$j9n?7@H^>y@2#xiHBRFY9;RUObvsadzc!@HX5WlHPoQZ!tQ7D{;Q43^*V##8CBI$3ZWhA"
        ">V~*~_U%K?AIaf2b2vlQCaN;tLW6o8kl#1)IPUT;lfnD9s?oa0n_r64?^MSFuX6~6<ubZ4&e94(_Bx+mg#E%`F(Pp<>fQWOkE5FS"
        "xJ`&;e(^`WS;~qd40-3MJ2P#g$_;ciW*dfJeMoM0^iiOUm33Et52x;HL3i+IBo7-*ej~CERF1*k5YBrooq*`?8kDsKSw)S2oH$Vy"
        "a8h`3^a6N|C_M*s_a`<}2W1;-^Jr#<L}!L11NNF-3{=6KT4#rG==m`P8y$se>0#j2!oQ6fcarYm03GoHOsS%y0YJ-Pi-{B9@ciJ~"
        "H-G^kSBSi7cXEL<qzvAlsn{46sDPU@Nq5j%$E}&gtpO+^M^|Y2Qj1zJccNCjEv8$gr!TukHK1FwY|40sfzu8vCN1m)KXz!-HIMN^"
        "EZ&wS&@03PyZz-Rx_R^A8l`o6{qrv$uA}$2KYno3e4cgfXVoPZvLa#BBA%#1EG^*;gAX5H=_)M@z;WdJ2h;tF;SzGB%ji~7oeg$|"
        "bWw+J)Dpf)c>Yi8?k%h5mTKr4MU;q`@8%T70Dl5}+5A1v8ZA-0domX6D-X`ECUo#MVmvsOVVy%C#~Ml3^a%F-2tFq#`>ve`x+6a6"
        "ma5u;OBWUX;g9z*o<AwpUF|QMsM3l-;<K@hXM*I?>m4Nq+jG#V$JU>MJW$cRQ43&JDv$n}b{PbA)^SV!_~@V@Ali1|yM@Z<jrzkC"
        "eRV+<eEW9r9K!-M>gsP<Z@ALs2wiWGo!b<zOHlla<xmMUjF(veQ)Q2XS1kH5<BJIbNA1RPqn<+$8ss;gG`WR2bhkQ~ZOgHnfwUom"
        "z5-zlcHF?(EE)H4*2II?auXi=#nI!)y7=L{KU;jqTy`1}@WhCcVONGa<BfUvguS7}W!HynK*=lh(~Cjuq$Xjzs|3|!G5s{_u13-d"
        "WFT#=1iy=NkppfieQoSKDl0ei{loBWhekOGD^qfP*y0!`!>@;kmOQl@B$W!gWz00HBH7hE_7_=)vx|P?%xXj#5|j&jz(81!0lJgr"
        "^|Jy?edk<3t^bq-($CZqLH24hGMJ{(7M<(&V(Fto*(s`Hf?tAW8rNk+QK~SJh!4gZ)ITKiW(e!3r+v|B?GZ7kRT>jd_VmSgk*mqF"
        "R!QKn;9f}%)nX$lTfzx>Rf!_;Cv}Mh9rgmvz@xljr9bJ5MM-uUSk)p(G9#U|V`lzj;!Z?mT?8XPlPmk;piibH{U=LZH;xn<ml%SY"
        "2iQXSVAUOlN+JBG>Og2f+A_kRe|%Ap3tmtm%gmiLyyQ8>RX5q#h<9BT>L=_i^c+#-c}{`nD~daZ3p;&g<}B<?i#oLFV+uPaU+jeW"
        "lGK?A3<#rEJideMEAhve?~m)rvXGOqO%1>QmPAx4YzE#uNrm!%Rfw-cc4MSHl+{qLaYoAt><4lxm&M|%>(n+uNMEQ)t=iyf{akl("
        "*gnAzX@lQDo)y!R?CDs?Nr|*h4vHe|+B9oI%CT_*CEAgd+VuJN{xsnLEOFulkB70S87E#X=IP6C;NDh@bq)D1e&;c#x(AktnD2D@"
        "uh>NZ2r=sil&D$-1&UoROxti&u1K7e`j+1`Th<MxLo=yk;bqEqTNRd*es{yXQO;-z=Fk`O!^aFWXGr8Us@Ua!xw%D+>v1@<ZR#lL"
        "n4FarC)MS{H2w6B`&N>M>TJmlaGV)yq^h*N@tKUGFWW|yT4`~{L#-pA0<QaX$Mdw%Nbz7Eslv$UxpUnkqHTQ4Yoc8dZ|f32%9%Pb"
        "5d6ZEF`O(q5=b3U96Pl_J&Pd;Pj|MZ7744e@<-wt2dv17jcH<r{&L0XHlTiQ6gg2W=OZSl_sBrCX9p({6MJvUu-I;v#;2EMb7$nx"
        "lRrZ6ChOBpjY`U|<}TXL1`yGpGz}aWexHLMZ=@^9<p3_svQ!?|kJBgvQ+#=H@`v+@CD0X%r4DmpvGLWkP65@AcDUzAc%8rEk?YX!"
        "btt1YlWANByoC4fV)PM=z9bmY(aj?U_d{@~t-By-CEG^>(a7GlyrxcO|1VX>kE^PqpMBLG>-_fgVWPgL>re@3y31oKKKdDnYOmx>"
        "?QrYZ^lV0@s^hCAs`PuLy%|*aw=^?u!*Hy5Iq;c~OAZFipAYdT;e8ymm7K;w&`jcT1i@p_wij7W-oG4R`to?7>fsjXXD^e0NL@BQ"
        "fUWIq`_(>oZQo3eLlE6PMG8=yHtw4~6@mRjnCbM-=FeB~kUE^=z5VgSPd{9Lyng3>`LAZ4i~@~f`Pdg(qePZOkV=wpEOA)B(HW3y"
        "?b+O{XOvB8(b_j!0%u{>xQ`|hBQLz#wW6_q6LrW7KkfPL)Wf|zmTl|%Cs)_QezV2bT|ogf9QO+R{}z8>5Y=4)98i|(K9A_QpJaW-"
        "1=h1JuMYn3NeRc1flg~gV-Z1K@IO#X0|XQR000O8001EXv^F2+69@nRW*h(jBLDyZb8=%ZVrgb?V_|G(Wo>O?FLZKYX>Mt5XD@JX"
        "Y-wY8E^v9Z8BK5FHufGM|AAF6($u4ATC|7k_%Mr2fdSGsXlHg0h9NK(ov<TODoG`i$@;(Vd!$H_v}~t9SC?4g`~K!jmStC;`-X40"
        "CR^I_=85zjygm^sYEm=JWXlCtnr}$gYu@rNOp-g<4ZR}UP7={+cH4EI$nQUWD6=d(KRZ9;ZQn^vHeLVZUs|f~gEQU9&3!2Zp^6BB"
        "2V2>-<d$!{4Q=jN+tN}?3arIl*|N5iPgO6wTSi6#H~4bh{r2vf3r6JvxQQ*l8zi%EmBU8b=VvwBf;L+^G}@ma;FdQ`<;CUs8G&ch"
        "wQVQJl04gYe6j~774JlLNiwQ66B=HsrjxX?mZ}~uvPA$dsrUrI7e9s{Y17|R`1<{K;n%HX><g<NJ6S6@{2&K5bKEy<<`C#9{GR>M"
        "aiN_AJNw6JhN_I8&{?C0xgdF_p4yga$v4>o#GBrnE$|ur3`7KN*(G_-8VM}FW3PVzq(>^-p|6x?Jxc$>&ISZPS`JKXx|&I(Uo-o&"
        "*iFRw*2%M6MiK-9OhT-{g#1plVE=555*-(1BH>K2vdq@8cky}k3L~|w8SC*!Zlj=B#88>V2kQ1mdTgSqnuNSOpk0b*)dO(*BREcc"
        "xpG#lIstMdikwzrQXnQ2g6u2t6Kzx!cV6CS@70rmv5TWqGCfG)6`L~j0XWxGvmixHiskmtg0#S|S^m-q=EBq&c`RGy9n-nfw6Gvk"
        "ph2<1(0o1^{ENgkyC4=pB*n<TX)~}ZDLa{GAH^rpJqqHaeLa>=%dHIfhr+pfQjo8fM@kBTTnf=_zFQ=l@<(O+6~re7j(8RX>!qvj"
        "4;O?hqJ|#JDXR}j8<nU6QP7B3j8XxEY8Am974ttp_sz}qmA4@gjMa)zqF7HQ!~&?{!lHKk0$KG_&+<?VQ+^9C0UE#yy0WKh@|Ij2"
        "70Vb3VM#kEO`z>iAfdSC4_MqZyJIqxBA<vEriu%tso2D#*(OjdNDd?xWZQI97sZ5X+o@@lC0ARphjs*3Bl0>1PY%i3CHd9@6sPqw"
        "NM1q9&a{1}sn?Kw?%4+Iza;sT_PeBnN0~zy)D6hAnI^C+$bUnNJqM*dr5WU=rKrG=qg0dm@ndM{u|_SR`4uK(yaGwgruEJM2cpeO"
        "@)8T**etV>yzv}J-8mm`?$biy+UEil(vS=atCjQdsMNXyH?`Buf=>hUk1e|3A%|-<gt}ZTl5bIc4gv}+nM5xVj)HYQMjL4k8fgMB"
        ">EC;{*=FyBNojKZo6*@!7rQbM#bmY>HO=Nx*SCL#*N8L24|tir;+SB2xr{_P6yynaN2nZwJ|u`@M~66l{sBRWb$Oog;o$n072-7t"
        ";p(|$e-2zSES<N|DH<4yM~0zY4L#F-BLyKWpi9WyAV8{4S!;SN3K%P)AfHOjS>?2Gb}@}2$)Y5a(nLy=X<p}_jf*H;8zAB~paFwi"
        "Vk*d{`R<EPVCv;TFC6lCP=T;GruE(9FW@3mzOS*>>ZjBC{O-s)xowS7#E~GrXr!5Opgx|K&fb+y%l1wk5{{AvC<v&emR2I98{~q^"
        "m>p=0(Or1iZ%&Mt4Sk~p6b!cvoOtNA7qeJqI8;FtV=G_AGeA&t0cU8;rNzdEeUD2lm_4wog;=Dw(A2UPtW<EN#nEQD*RS6sgGn?W"
        "8(k-x-VVH}CnMQ*@Hdzgd?z^zZh5Qq>5CkW%~RJL=X%AMunowy=&ucxy24Se#bttmTX;R-!U4pC@$;Tah@~5MOuAc)SG54fn`Wrd"
        "YQP8TnJUvB>1m>n$_%mA8fD5eFA`5;miXh+RZcJ|trHJIoI&FpSR4p-?y(EQ9TY#ERCdOtw-9QR!jDve2ZokOGvMk+zF|?mBkhvb"
        "bslj|x;A_*AvC2{k6hn-j|J5YXpXa;jkk32S<l_N$ue~F1OVOTFjL3LoHJ3|(>yE91?L@xj{W)1tVk2ab9Etiisu-Qe+y3++aXVt"
        "HsqNJ<D>AmsKCYESC`|w@sFu!`BufHimUsWj4LD5x}<#%+T?i>@y`8zKeJlL+3~`>z1)i;cSTNz&izTpNs3tM@q7Y&6n+e!=HV$z"
        "_vVz@+(Y?}v7YhIRog>1ojhZ7a$bp3ZJfH7d-_{m>1NFNCl=v(-xwL9XP35e%^Vvm;I3q4C)5g@jc$lET_41(@Z6va8v+Yby$s!~"
        "ZJgdKe4PcuiL!bjY_k2C5Tb(=HQ3mN_c~}OFiSpT?JJCG{nVzz0&mt9aV0;#yZJR;W{(%MP4~cXs%hp@d5Y1oMO)mdM<@NZ(ukjr"
        "o4uq-k0e&XMR;yC=r)5Z8$+qx0@sXS?!z8r;+PTqs8_N{b#&#>=6#9Hs>uUyjD*G*UhgeVYrM%BeA}d-72GuX(tJKlX17PjTrH}t"
        "?7lGE#=8`aHdK2G++1HBM8Q?nzLkJZrJulwxGA**O7;Fj1#{nhC$nikwsLF6b$U%Oi;B!Gu(JJ?E<UKYZ9RO7BRp77;pG40VyC^T"
        "rGjsn(qlxuJd(!pb2O0RO44x0W)+?6yZ8SVfPc|gLvKdjG4sqd1v3#{pmJ}?1wls6a#&;FHQ6K_s}=^_?aV|c1iPcruFUXD{_o;N"
        "_hQ)MH4gCsYrn;rHBMj<N&S76i~OSec0qC@+c)H*fO~~soNOIU6O{m%G%;JqCyw5G<dx?Vnnt_n#Klp7i)c|u8u0V8{{c`-0|XQR"
        "000O8001EXY=D*vi2(ori2?usCIA2cb8=%ZVrgb?V_|G(Wo>O?FLZKYX>Mt5XD@JaZ)Rz1X>MmOaCvQ#L2KMV42ACo{|Dhy0vo%~"
        "n-7H?k{(KzvQ2Xh%2?waGBa{DayQ#wU(KeQk`kYUZN2yONnO|7efT&+A}p)W1bdX}fI_ec`w*#zEyMlp`YTY;kWdLi0Tr@p0xW?4"
        "dWFgwwePyFZQ8~QCwL5=9mbhb!rpQAVTO<v7KNMVg(M`(`h6ky-v7m+_Z+8l$E}%~(Ux^bI2~_9SgKRoOytjjALv#0qL}dbpV0;^"
        "hSIY>C_hmyBZRCQP?VY8yt{;}PcQ|iW7~i|HrDN0{YN-ZnD<%+UJOb2O){X<@C!9?E<>T{ufE(5PPh~+s$-l6qnaQZw(O354P8Zd"
        "beil;ePdIQ3&nISQq2O9@~nH~oDFx3Mn_p-A$%%OXhAW>c!WytMUuRyn#NXW9}_7_>EZ6c@cG*haJZlg8psrF%bC@dSk`3gQ-()t"
        "VvaQRuQR_Cf}zm63%IBU)NP>K;>k4eM6vRtSx!Ley$JnWLp>31*^SkT@$bGup8M{yZT<jIO9KQH000080000X0O-oKpTGzJ052T?"
        "04o3h0CRF<FJftCZ)0I>XJu_|VJ~!YVQFq@Zf7rYWpHnEX>@OLd0%s8bS`jtrCHr?<2Dk1Uts?SR`(*cT3;Nrd)Ebg4|m7{_t0X|"
        "4T=IjLttdeCZew+DaUS*|GhK(5-D5GZmWkxB00mE;g7!=(n*s1!un2f)rk{f$G#JasP2)q{CBE&*Am)pNkdh+XWMsuSM%~jK7YI?"
        "RVN6oYa;oPlwDi#-5|7pTqH^I_RZTjmFODMQ?;-81L2L<{0TmN-vh6@lGZ!cG<0EfxN&#CeY*SQzdzmmd|!P2boZanGZISBQb{Lt"
        "Z_H*}@Q8A8cce{UvrD2|s_3Gts+zZ~kd<;8(zkC2JiPz~f)#Ab6>oQiU}Yz^GB>@lueqA*Q}m%`MFl-f^n$kvI<i?4#29*Ofnao7"
        "$ZilND@6AsXGNnmvch!-!UraJ#o4y7Bbd0P10u&;u<}wADsV4ozAA1JVI>_9SYg_WrrWYQCtFt0p;m<xD|*pYyk-)y@b(RKJIsOC"
        "J|;G{c2OW2ImDioIbm%HdIjpWFW>scqUH@(tM6}qeDC{h+dr2--ok2pF#33fDi=GZ(nNGMOEU7!iX_^^#E#MPQZ5<!mDU5h6QUDo"
        "!Vd7E1MgP9K(w@7;2^G1k{K-sRCZKkg%hq5`<ZM^ud|zPC5%u5KsL2)3-dp<5;>t;00)GC)7mVrT-}R-Wf^*d9acgRR)f)L=3Nat"
        "1Lt5JnjS4n$2NR`6x1ME(zW%8P%^X{Qe?}kic~`lYX|7q5w&N8w^E6r)HocZYjt*5(-;41lX!;jOfUjz7Z7}`;poLBYyX^iU<8-}"
        "Gh3sK;)Y2{cMOBSel|$>K$1Z+k>tcIR;!@NO#ZU*6j1^5tsKM_wD0RvQSK>#fC2AN#e$f+Npg?#)kg}>nI+w^mKpLa%q)6U5z?AX"
        ")SkDG;Exe8N^O9^+4`%EfCvPhQYksm7Wf#st5-)+0sEuqk^Ym-%O)R{P5c-;E;}?Gb<wpv^38-e+854GyD!X$i`zo$%YrCf(}VpB"
        "0XX4(n!T=$S6<`#1F2b?j;3ZCWRdavDv{V>Hp&m(&~B+XIodGIV6&fWG8Or-@dg3;gaL!(JbtZjv{AR%isP(e_%d(ncS}8BdGr3N"
        "xdAgr;jv_&jNw5;176F>zhZjExp9p1EF<s85(Y(l{iU`T8#>5)Roy)~D}-T)%QZpjxrerpX!$nT<RKB~LG*GPrEJVqu09Bju*B~w"
        "2&i*>xd184uBgk%MM<HavaR3daUuv3nCUprVoo1UYMXCwZf^Ye!`SGoU~Icu{v#jrpDSD--Pei<+#EA?Pe38l@+3t^Vau4NRf3W|"
        "Egvgn6G5M>9m^YhqTOvV3ngSKY!cS973$PuE?xo42l3SuR2-IxE<6R$x^p1Aij(_2mqZLL@DT|7f8YZ6PfRObhyohZ@FQfAn(Zi*"
        "G*<n>0qSq$qhRcJ21y6fTg`VI2n<rm9t$5@icU&j5bLr>k_QIHGO`oh(1T)KIAUNIPUF9+WLRcNvZn_IoC|dp`iF^2(uR=(mv$Oc"
        "LiF_@9Ud$$rPZb(1&jFnf-p5Ws0gq-3`qgaOaQ=G7%i-TT>C<s40zXpfaq#<!C*hd`dnrM)6UeM<IK4({?o}GaGV-!+1Xiu$DC^q"
        "HgJ|sRug*pF5WZn{loIsrh9Xg$=ypa!ha1UfYH+_15wsJ1vm?0p~Ptt85~~1Nr^+<hc(C90cJ8jWP{m?cluxje7VO@J=1Sd;D@<N"
        "P}hxTt)_VV8F2ok>TI;Lnej2RMPD-#G3CXJId5`|iVLc<xXD5Q%O=)tF)0KC*s)|#Y>s3CZWD$tleGj+lu-Mw;P59>J>c>*4ryy6"
        "fSYF0qj}Ku=Hw;=*e)Zk9wh?)$5YM$XTNkTji}r5y-o%ACyx?H@SX=L#J@%C?GIk?QZy9mZzk0h7SP8JFLaFv{Mim(Ld}6ES10lE"
        "ud&VKP!ue@K00am>rIeFo|jOiDQS6DFM_KuG=q%nq1OYcARU(}WM5`V5gCX%QA<2p!vt*S)D!A*NWQ$OzM60}Y-ULZpP;z8^ti>C"
        "AK@_Zs*bN9k3jDnft;U}U*!&bvR5hPi&+DoSD6DJ1Gs=ROv5Z!;Nvmr0O0^~UVQnX*XyV|IK-U{U;G{UhE@tI5EQlrB-S;qRtom*"
        "^QbieYy*Gcp))|BEo}-sykmhEgg!BxW{(#j2E5<3yo7^Cpg@;b#lEYz-Jl|ITTvSO3KKU_W7#$w>4?~rb7vA0p1vXp2L>3w7L(J("
        "S=AaIr4w_4nNDh6qHq9dLx<+>pgGzT&_B0Fl4u*iz`wFBl95_XIRN^eohil$Pv5zp+Dz&@UP6T~U|Q4UI?CS4k@ayA#8pT2h*9+@"
        "h8l4**}O3K;fq9<k!WO_J@WJdj+rg0K!x!%&*xkB(OjcEZqB=K`bV7|BB11Ms__k=j;DaG)lDt!<-Sve+|%11-eZ%7e=;GCm-F&8"
        "u4c#nb0lIg!Ygcq+}XYMBxS76U0kP8#B*|7L`UqSD+r44uzg(E>kxe{;8%$&*2tMS>b__x<8k4=qILEq=BafOy1UVgeCDtl9A7Ym"
        "$jc)z<zD@GbAVUt;WbL-;%F|)!?Mnw<?S1gTR*rR=yz6ph0xGrAr<Q-p19q$Lf@5pXt1BOEQe;O_2t~^``q~=hX%YAU9+-vze@)d"
        "ZFqf(3ouV4c>fyQ>jHylKRkuN*&EeOYpX<_nuaOC%Vd<%+@m-v97x<GPrCHgk_jCp<Fd^D4zZg5sUgkQ#+QvN<SzYIk=kD@Oq;Ly"
        "njQl7z#$u*COM6Yat{D30Sx*&wS|EPHd*9pUSr1|5@us=S_Ai6IP-_&jKP?)^h?C$RME*Mok4T7zkO6(PBky_wy@Bi{0<b}BnN1n"
        "A~r@XKDrV&@C{FhQB;HW5ybv2yf`*+HvhtF=MdN7$JETx`&H!obRPuo<Gu7BP)h>@6aWAK2mk;8ApqL}Oy48|0018Z001Na0047x"
        "V=rQ9W^ZF*Y-eR{ZDB8Ta$#w1X>MmPbaG*7Ze?;VaCwDOO>5gg5WO4x9|oNgP-WAm7ePu(VwW6ZLSpDqjCW`4rTd}oM@T|{d}n3Z"
        "F2$ibNUPnqGxOfOsj6x{|JZ`mfkc9*<r-wr&lI3HC}`)6q|m?<5lAoMZ2kD{Zgsc%-dNdGRW+SVCn|Ug5)%r8F;WC=dmqwiYlQTe"
        "c1CXw#}E8_yuQ)O3p87@78~K6(%S=W@zPV!mRx%3Nu0w+fl6_qhZ{@6lmo4NdL)#uix7M$nWmGk$C@rU%p1Aub;xu%oeX=0{mqcO"
        "olbxsLl~TP+b&?D7L@Ur;5V$iBe)(udz{rG_-=zq?7FDm7^>Y_;5MMt%%nphvdspQ_O6SG`uOf0i!c8{^xQffIZ_KTg|i4!iGvgx"
        "+t+tdiBRk)bb)?lwu5Y8<Go?X5>2GNf|3+)9PPTjW4fr)#r}Ha_O$=Bs~%j?S#(-aOzqJ18ulG-PlDIrbLw*{QQOfPm+;}sP~F%K"
        "w%an7#<tAZD{ZI%BAH^`p(fn$UXKASB?_?|-z>_TX^Ee`XZ@N&qC~r$f12ISuK=Bd`}x)Ee*PJ3jtMpdq1c8O$_E19CvCZ@VOFRB"
        "dqA=!8d&dmBuHU4X%v~O21jLf3}(kUBj7>Q23FY^*uAnke^oe24VEx6GdVIV367-q3+5N;thGn^2Q<%eRYUco#4P33;Mil&oWM&m"
        "H9Rbq>-t1#V1E*a%Zb@iGp6LZ_ybT&0|XQR000O8001EXAT5#yr49f9+bRG62LJ#7Oi4pUPE$qQT3d75#ua|gO#Z_TXWEfwAlh-0"
        "v`L<1OODw(l!uh-$>S1OkXr*-WET*{{`#KF-at~aJAJFilUNY;?Adeq&Uem2h%fxP?q#`C;=L|a=hU~~JoD;5m37*5;??|RA^tAA"
        "QQCue_44KSPp{eyL;v%O7yJEwE@`8=vD+67Z{}Wn`wcC<zg_*d7T3!g@#b!Mv$$W}E!X1h-Aa61-xgwZ`(br=^YINnF6gbB#rl4="
        "c>NKdcoE;t#f_?Urw7o`&HYpOGxH~#3Ad9?BU&XpF~ErjWm_j?SBcVe6|WPuv0`+p5LWfpRAY&ciy#YMuCyDh-i-JQA)TmjS5=A4"
        "L9A8DQoe%?tQogE@slt${Gj1IQ;x0bhEvsyJ<+W+{lV((ZV+bQDJwuLSgMDEkmIm3R{xuJ41&%r8Fq3Iu<zDNSlMmqC2xy~4r(hK"
        "aZA!q^cXulB^^NtS(2Qf2-pJN4RQ>;;g!_N@$O*P!J4KJ(yH)7Ly8x8B78b_6)Y=F+nO$j@^6a09(F8<x15W&#*!xeXnO<klZ+c("
        "AAL3RVrFz2Coc3QuQhvR3vi<aZ=o8x<9`=oFrt*;eS9y7;op%JtY~E?w+g+8JG-&mdDRNB-zl;HgaK_VN%+J%dyRnwVlFi3LSAw^"
        "-J_^l*I?3K*%D=6{P6Nmm$aV&6S4K9q;YTq*d9Fr&a%n{ksx!UIxs}3!J!lJnWnkxe>dYyT)=Almz`ba-iN<vm<L^rD9wsIPJ#sW"
        "2pVgLx`6Jjc8<cILXJ~{ymKb%*05m-ISKjJ9tT}-RjsTA0rcCN4F4H7X-%c!e3G&(dRlj7Gm=3dD@A7p(dbsA0^m?*>S2$O=4eN7"
        "P6bW~$w-od9Dbt+kWlOGXz2xTPopN7a<};xM8&E4vODlI@XOdxy42RR@XK;1JJ3Al3q(lg@KqUti=H(8$683iHX?b&^pHWw(ODsl"
        "dW{KYNIienEd&Le10GLKJpna1?gN9nLqVL(t*W#X!=ay^{gbhuPk>?%4@ql+Cycw461odVNeRIQ^k-{j1?ci1bt5-Tz=RC03dC8A"
        "#!~vQk_jsU>VYlb?Ff5pFL+y%apf>z^&v9_eS?e(ID>kWZ4Z0HO2B=HdR~dIUiZC%y&oaF8neI5O>v{FegJDeD1laSv*TF6ZO$9Q"
        "TOf#HLj>hF(xDeSO3n&5!DN6yVhF>%$ZeP!`<*U#IYNNf2B;8_tX4f}@+rnN*vIFMP+(_c!%rZ?`#aCBAPwuFau77+1PR-jhH?bf"
        ">8<Wyn^WJPB6WnXdIAZBI5wfTAV#)#A<6d@#i~~7l(4FomO=(?LnmpKRm}mCqWet7+d$Z0z;v=zm%;fO(8<=aqykdpMKoIWL@l(C"
        "GIj3#H;D6|37qx(aYDo#&bN$a^a&WMZBzxNO<Y6qTzSS4MH#jbErS=HIzW+S$bcomz}`(nbT@7QPCT9jdO~qV%9DaVFK8!%_!IIG"
        "y-oG-*;0}d8iZ@wA47kmz@Rl);~A3kb&NpF;;Cjq7;}OM*RV=8kdD>>NEhgejch2O_7)d*L{Y}hyFg$<Of0XGvCvk7ONm2HaK*D_"
        "D#2O43;a!50ZQryS2sXFL0nePB4cs~H>lQ4pdE_xsIa({R6_kL90gbxnbSm4mRr8aVPz5wnaR<1z=_Mz5f7j(Tf$J!IX@8wW>rf)"
        "2187b8v+wf<9c0=W^|Cst+byJ3~eG&k(()}w;k04h)MJmS@&#6B6`i1V00<+yq(X^CG-&^k4F#b_+@64n-MXyJ=$07K%b3*zyf-y"
        "2;V{Pe4{KQo%&;hKx%N~(ty>uoFhNZaxp;DtGW0U2^jADCK@4dFtHxF^!l(p+XdxGmxHx}8Y(g?3B(l8n+P8<2OyGgQXqr98o)d;"
        "5&;gIYOj%xbf&wa5O8p8{P7C7k=<g?We&0#4p+5R@RJ4#_+U!J_tR?cJ4oCzbZW2!GOov%J^{ZB=KFC2i@}x<w0$EX>f;koo9nQn"
        "C!TubJ>R4?hzL#e`ee^D)t<m6_~|d1pFT*$;{OjPUBEik514S!sSJTgfr^ehsY}uGaWf|Z!3SwO`JjjmMwO@+H+7AK5UPV}0G0W_"
        "05Zl7?4O7Xo>qJMNXQcphh|_`g?(k;H`uJ24xCS>Mm+P%m5tP3bbdEKADEJ)<R*@2+CeruC#|Mrtt|jjXoQrGbumYR3wH^9wCR+u"
        "mVk*sT4EbO>y8)1GjWIPYcZUF$>3ByzsDZ^5$fPvENTpc)IvExFO2W#nn7<_3AvT{YXUys`(LD5mYBFT&Rvn!@Fb-fk!A7E5CBp%"
        "a_>ex;AtDR<>~-d3+iW_;Rq_913fAhth)2PdM2n$E;@vh1+TQkX+aM2wP{F3S`}JZpEaR1N%`i>VPIZdH85Y%hcRZP3$(6+ry&ZX"
        "i9sB;m{ov=@8@Er@<G#_c5me&q1`bs0XFKuf2I)i3>zfRAisp&MgTqvC}chGf0HWqwD0B`{{*@fsedA?WI@&{#qO?61DzBTfB<Jd"
        "r#gHgFZsA5MBNs3Mise70S9Y1Cjwc{lT%}fe@;#-sd5}QAHQ&!joWUrZFvrqs5kasI0WDsk;QO=mP2<KGu(N)TmuBY7~v=s+y5=u"
        "M6sab&C6_as|HB9A`tCt4nQ3RD0O_oeDAnt3XTGEE>+LMM|**ASt(>#MNT;=VuzG~{(wBnIHS=C*5aSU@)%Zw5K^x)M1Y|L;Yrb`"
        "oP&1A${HW$w3;8AuDV1ti?QI_M)dk@d3V2fb322Cd>qK`n2CNHWa;@<dA?=9a3*z5>`QLTWrfyNf-hx7EozEDb=J6uRT3x3xg3vn"
        "gcp1kIz{m{*3IRdH}qNSQrLlzMoEW_*yKvUuSp3E3<|dWIVdWFmdW7BO4Imq&uZ~Y4)l|7n`HWQq9t^l&<AVkHdV;e4U8?$m|6y2"
        "n9a?6M{&kd^>Nw|%YgQgT?P_ub%p0VL=Sa1^Mj5X*@u$Q!rt$=fkRL`W9!WA#4xxQkCQ{_JW}tAJmyFtpJ{|*hshm4EGe?8@ZT02"
        "r93oqF+qE8rLReff?eRi8<S&ED~3~&s#0Awwt?hMA}1gXH;=)oCoo4=4zmz2#o5H7*%vfh%oXf-5{PV^r|(G48Y=Z7#B%9UfLX%P"
        "BrW#{O7Vxu)y65GMv^;W#b-!=HZDAKO~rDu>nT&8(Wl6AsHR@?@I(ib5By?YkwDx#pAaS$POiu%WjgY7990wd#(<hmcw&Qd)IA*e"
        "<p=7L{G9{$x`{Kox%k*Ym38Dh^$2^F8XIVmnO(=ENy*`ubY|1BY%KQVC@gCaaL?nZ5EH{qKHdJWx;oFuNYN~ESO!zwD&BpvfTbC5"
        "RlF{u%Cj-t&tV#GsWrs<Ldv-j%7jyu;;SdjiQF@OD<&X3dIeL)UTO>dASJ=UXA3oR>QR+B^b;OOqgb_-manpoTb6j`@FO5(U?~pK"
        "GUMKrAwV6Nre}9hXnuc@#~df#n8+4qagq1nAWhi^ygBecf?7VF`Q;G6LFpJWsm-<Ok5T*kDlGU8T!ST_Tno&=w0KdD`hcy0nUruy"
        "?~v52;1u{|_8W35xX;&#GqoX@>#uWhqaC$fcyC;bPrz2eIER=KQQOUdJ7DVGu|3Oxf!s-LPnxV1$#EVHT+;Re)y2`nNspVqj)T#O"
        "moITr2E|}@y%vl0OuW8cFV@k-pBDG;?mpg&PuHu}_40mkyB2q={2PJ0x8i#FyZFaqc>|nC^Bwdf4qjb$7)@AHWh2;>Pc(y-0lN;+"
        "CX>0Sm9l4$7c6?exPO0Jfbo}Ci{;zZV)^UsZ@0_)Li~2Sdh-tSynelSzqtRMV(IPTetEm*uNkg`kPp`@aNNiD*DLYi<LblR`j)FP"
        "-%~btg8@hAVOveN8*~fAP32MCfCH`Rtwugihpr*c@EwZ81jN}amZzA`0dc|;M;O<RAm5o%$4;Bk-e2?5s5igoJ?(}=6#i`vPl91_"
        "*?TQFx}i(_1(s$3B&i!n6_x^z8k(bmn$WH1W1Dby29Y%6qlZpy8x1V2RF_4(bT1~8;xtG4V(ecqm%y8YMsJAuk$PL4YorU3u<3xG"
        "LpZvnKc6d%sgt^b1D@!)M$^`Qus|-Bt=vv;3UGP&Iw5_h;Ch93{P_hqWEZetzOq1?!xLk?1oTn^aN`&cG{=Fs<(qq~@~M{Nwe@kM"
        "OeP&8>WutEciyEL*iPmS7tdbI2fgs*jp5MU8dL3cGnq(!hMH~qUgGQ#S;L6B*HSm5<w7N!dh8Mbq5^sD+Z4QPz&Orr!doi`0fYgG"
        "g#LK87lg!ty{sNIUC-A3dlJY7Z;0@zg_j?);itK{F0ojkDFPnj=GUo^X9@CYhg^S}y(gEH&tCBb!dmW(;rS|!P^VY^G|vTsS1Upo"
        "fIrevcBSHj^gLkoh<Km~RBeZ!yJRyqHf|Ht7iQD=u@-Tg7YIa1llfi;j)b}6+cF(tH8v&h%pUtE?n0x{$f}v-bVT~jrfc$xmq^k5"
        "4Gc|){bNMegsr48v7vNflL37iJ7w|UC(hVv>6%d+^Pkh7>{BQERH}v^hnE14uFeeSrELkC0u_&z%(8p5Hr>$q!89N>H1*i%@_e>9"
        "ol0*Go{eN@I-nJjVI%q9XJMJ~%cxC=<=f>AR^4-7?D;p+ozttAuYQDFu%r1r$c^~3c&%$w%H~(qwlde$@k$>ARmPF0Unds%Cy3$#"
        "Ux=UgUH(Ec0{cXtGAQ}sYT#?S^UVhZdYliW0wz+!XEro7k(1#~07W^Z4)IIhIlc#)wp`y{f!co2*O0NNDvST5iR5JCmD3MraIl=e"
        "@FdszABk3ASZuwuQWGV@g)x;aXhY$2-F5_k1iXh-?~Xqo3;lAUX72Q8u+@J6P)h*<6aW+e000O8001EX>8?v3)&T$jhz$S$7ytkO"
        "0000000000fB^si003ieZf0p`b1!9haAk67ZDnqBb1rIgZ*EXa0Rj{N6aWAK2mk;8ApknoanoP~004~(0012T0000000000004ji"
        "3<3ZEV{dL|X=igUa%FIDb7^#Ma(Q2KbaiBTE^2dcZcs}B0u%rg000080000X09`3Nre*;E0PO+*02%-Q00000000000Du9f1^@tK"
        "Z*FF3XLB!fa$#w1E@gOOZE$R5E^2dcZcs}B0u%rg000080000X0OyMqCVK$@0Eq$s04V?f00000000000Du8S2mk<OcwudDY-MvV"
        "a%FIDb7^#Ma(Q2KbaiBTFJ*RNY+rJ3cXKXkb8l{JP)h*<6aW+e000O8001EX<zpNT!T<mO8vy_SBLDyZ0000000000fB^>z003op"
        "VQp}1WpgibWpHnEX>@OLd0%sMb!2%jQbj>TO+_wkWKc^10u%rg000080000X0DCRhHsAmN0E+<t0BHaK00000000000Du7s3jhFR"
        "cwudDY-MvVa%FIDb7^#Ma(Q2KbaiBTFLQ2TaC2yHbaO8;VrFDvW;QlBG&VUhFk>}lW;kUwGi5V5GBY(bFk&${Ff}zaIXGo9IAu0s"
        "Gi5e6HZ)-}VKg*1V=*^5V>L7`YIARHP)h*<6aW+e000O8001EXyq0&G-T(jqiva)tX#fBK0000000000fB|s~003opVQp}1Wpgib"
        "WpHnEX>@OLd0%sMb!2%jb8ca9b7*gLb1yMvVq`I7Ib&ooGBaZ_Gcz@1Gh#V5H8nG3W;8W0F=8}iFf%k}Vr61DVK_N6Wo9upVK8Pn"
        "VP<7FH83|ZV>K>nb8l`?O928D02BZK00;m803iV9uKDKR0001s0RRAL00000000000001h0oDxw0A+Y#ZE$R5b1!mbaBp*IbZ>Hb"
        "UvqSIWO*-hZeeh9Xm50LFEeB`F=8@gI599~Wo9)qV`DgBVKp*kIb<_rG&MIgW;J3nFf?UkF<~??WivQ4GB+`1Wn*PIW-&N7I5jdk"
        "HZE#&Z*EXa0Rj{N6aWAK2mk;8ApkBb)t%u0004^t003zK0000000000004jiJr4i?Wq4t2aBO9BFLGsYZ*ysMZ*qBGb98lNc`tKr"
        "VQ_P3Z*+4nG&EskWHT``HDzKnF)}tdW-((lGGaC`Ib=34F=Q}hG&MP9IXE#gHexnpVPY~kFf=eYI5A>jVqq~kW->G~E^2dcZcs}B"
        "0u%rg000080000X02YYYLEiuX0E+<t0BHaK00000000000Du9d5C8yWcwudDY-MvVa%FIDb7^#Ma(Q2KbaiBTFLQ2TaC2yHbaO8?"
        "I5se2Vqq{fF*#;6V>K}}F*7x0W@a^HFk~_`W;J3sV=*^0GdMS4I5{>sWH4o6V>CHtVmW0qWH~WpI5asfYIARHP)h*<6aW+e000O8"
        "001EX%m)Zz-~a#siU9xsX#fBK0000000000fB_2<003opVQp}1WpgibWpHnEX>@OLd0%sMb!2%jb8ca9b7*gLb1yYGVq|4BVPR%v"
        "F)}$YVKQW5WMVL5WHdN2Fg7w~IWuK4IAUWrHaRmjW@BbCGcq=2H#9LZH)1g}He+RDI5{qAb8l`?O928D02BZK00;m803iS~JM9tR"
        "0001s0RRAL00000000000001h0df-n0A+Y#ZE$R5b1!mbaBp*IbZ>HbUvqSIWO*-hZeeh9Xm50LFE%zYFg0a3Wo0vCIWjUiHexd|"
        "GB;smI509cF*7%1F=jb4V=^~oFfce|Gcsf~IXF0FVP<4EVr4XAFgGw{H!f;(Z*EXa0Rj{N6aWAK2mk;8ApmubL{Q)W004^t003zK"
        "0000000000004ji*%SZ(Wq4t2aBO9BFLGsYZ*ysMZ*qBGb98lNc`tKrVQ_P3Z*+4nVmUT6WMMfmWMg49V>L54F*z}1VKg^4G-P69"
        "VP-fqGC5*5VKFu|H8C(^H!(S7VPi04F*G$cVlg*mWMN`vE^2dcZcs}B0u%rg000080000X01e6!k>CIT0E+<t0BHaK0000000000"
        "0Du8N761TccwudDY-MvVa%FIDb7^#Ma(Q2KbaiBTFLQ2TaC2yHbaO9dWMVlvF*jm0HDP3AV>B`_F=AvnF=IC}GdM6gI5ssnW@TbB"
        "W??urI51&lV`DWlG-fwsVqr8fHexq8HDqKiYIARHP)h*<6aW+e000O8001EXR?pkzLjnK*7!LpdDgXcg0000000000fB~l$003op"
        "VQp}1WpgibWpHnEX>@OLd0%sMb!2%jbaG*7ZeMb5cXKXkb8l{JP)h*<6aW+e000O8001EX1iVedW&i*HYybcN8~^|S0000000000"
        "fB`ZZ0047xV=rQ9W^ZF*Y-eR{ZDB88UukY>bYEXCaCuNm0Rj{N6aWAK2mk;8Apr9x!ab7(004{<000~S0000000000004ji(i#8&"
        "b8=%ZVrgb?V_|G(Wo>O?FJE76VQFq(UoLQYP)h*<6aW+e000O8001EXSzRm5PXGV_SO5S3A^-pY0000000000fB~N%0047xV=rQ9"
        "W^ZF*Y-eR{ZDB8AXJu}5FJE72ZfSI1UoLQYP)h*<6aW+e000O8001EXV(>{Ma{&MVnF0U+9smFU0000000000fB`Nc0047xV=rQ9"
        "W^ZF*Y-eR{ZDB8AXJu}5FKlmba4v9pP)h*<6aW+e000O8001EX1(i)TNB{r;N&o-=BLDyZ0000000000fC1ei0047xV=rQ9W^ZF*"
        "Y-eR{ZDB8Ab8}^Mb1z?CX>MtBUtcb8c~DCM0u%rg000080000X0G6X=OydFo0AUIM04D$d00000000000Du8zBLDz%a$_%IX=ZO@"
        "VQgn*ZEayMVRLh3baO9fWo>O?G+$*hVlHZPZ*EXa0Rj{N6aWAK2mk;8App3na){#s001Wo001Wd0000000000004jijwS#Ab8=%Z"
        "Vrgb?V_|G(Wo>O?FJW_YWpr~dXJu_|VKiT5G-57lb8l`?O928D02BZK00;m803iT_?HYs~0{{Ro3IG5v00000000000001h0k$as"
        "0CRF<FJftCZ)0I>XJu_|VJ~5Gb7gdMFLZBfWo~JDWpZC_VQy(=Wpi{cYIARHP)h*<6aW+e000O8001EX%3W!xjsXAw1Oos7ApigX"
        "0000000000fB_&a0047xV=rQ9W^ZF*Y-eR{ZDB8CXk}w-aBpdDbZKs9E^v8JO928D02BZK00;m803iT}!ltQw1pojI4gdfe00000"
        "000000001h0qrgT0CRF<FJftCZ)0I>XJu_|VJ~BEZf0p`E^v8JO928D02BZK00;m803iS(=l*9e0{{SX2><{c00000000000001h"
        "0iiSi0CRF<FJftCZ)0I>XJu_|VJ~BEZgg^CV{~&aaCuNm0Rj{N6aWAK2mk;8AplhcJW_)I006Q8001EX0000000000004ji3pfA("
        "b8=%ZVrgb?V_|G(Wo>O?FJxhKVJ}}_X>MtBUtcb8c~DCM0u%rg000080000X0B64)>bn8}0I3B403iSX00000000000Du9_H~;{1"
        "a$_%IX=ZO@VQgn*ZEayMWMOn+FJWY1aCBvIb1rasP)h*<6aW+e000O8001EXrqNBo&;bAdX#)TN8~^|S0000000000fC0xl0047x"
        "V=rQ9W^ZF*Y-eR{ZDB8DVRT_HVRLjYaCuNm0Rj{N6aWAK2mk;8ApqmnhCvMj0074c001EX0000000000004ji(LDeFb8=%ZVrgb?"
        "V_|G(Wo>O?FJxhKVJ~BEY;0k4Z*nehc~DCM0u%rg000080000X06s{^4af!n01*-Z03ZMW00000000000Du7<K>z@Aa$_%IX=ZO@"
        "VQgn*ZEayMWMOn+FJo_RbY*ySE^v8JO928D02BZK00;m803iV056dyM7ytk=Qvd)V00000000000001h0V7BN0CRF<FJftCZ)0I>"
        "XJu_|VJ~E1bYU-Jb#h^JX>V>WaCuNm0Rj{N6aWAK2mk;8Apm~(S)?`y006%n001Na0000000000004ji7GVGYb8=%ZVrgb?V_|G("
        "Wo>O?FJxhKVJ~E9VP|e{b98BAb1rasP)h*<6aW+e000O8001EXHbN|uG!*~<(?b9NA^-pY0000000000fB}wa0047xV=rQ9W^ZF*"
        "Y-eR{ZDB8DVRT_HW_4_AUvO`4Y%XwlP)h*<6aW+e000O8001EXSzqrG0}KEFOeg>VBme*a0000000000fB^!30047xV=rQ9W^ZF*"
        "Y-eR{ZDB8DVRT_HZ)Rp}X>Mg-b7pidaCuNm0Rj{N6aWAK2mk;8AppTi@Igxl008zA001BW0000000000004jiM~eUeb8=%ZVrgb?"
        "V_|G(Wo>O?FJxhKVJ~oDbYo~`b1rasP)h*<6aW+e000O8001EXJCxc_UJU>M?J58OBLDyZ0000000000fC12w0047xV=rQ9W^ZF*"
        "Y-eR{ZDB8DVRT_Ha%FIDb7^#Ma(OOrc~DCM0u%rg000080000X0PCT}PqrBV0I6I604e|g00000000000Du8+pa1}Ka$_%IX=ZO@"
        "VQgn*ZEayMWMOn+FLGsYZ*ysMZ*qBGV{dYBb#pFoc~DCM0u%rg000080000X0RI=g%=ZWY0Hz!O03iSX00000000000Du8)x&Q!k"
        "a$_%IX=ZO@VQgn*ZEayMWMOn+FLQTgb8TsKXfAMhP)h*<6aW+e000O8001EX&twv_wFdwI)fE5$B>(^b0000000000fB~b!0047x"
        "V=rQ9W^ZF*Y-eR{ZDB8DVRT_HbZ={AZfSa9bZKvHE^v8JO928D02BZK00;m803iUS1~;4O0002n0RR9e00000000000001h0h`MJ"
        "0CRF<FJftCZ)0I>XJu_|VJ~HNVQh6_bZKvHFJE72ZfSI1UoLQYP)h*<6aW+e000O8001EXT*TlVVFv&J^cMgCC;$Ke0000000000"
        "fC0(O0047xV=rQ9W^ZF*Y-eR{ZDB8Ec42IFVRUJ4ZZBnaVQgPxZ)9aIaCuNm0Rj{N6aWAK2mk;8Apkd|_s$v=007rV001Na00000"
        "00000004jiaMS<*b8=%ZVrgb?V_|G(Wo>O?FJ*RNY;|FDX>V>XXLD^hYc6nkP)h*<6aW+e000O8001EX`~+qRt^oi5`T_s|CIA2c"
        "0000000000fC0?t0047xV=rQ9W^ZF*Y-eR{ZDB8Ec42IFVRUJ4ZZBwIa&BdFb1rasP)h*<6aW+e000O8001EX=IZSZqzeE5a3uf$"
        "EC2ui0000000000fC0Yh0047xV=rQ9W^ZF*Y-eR{ZDB8Ec42IFVRUJ4ZZB?ObZK^FUvPPJXm4&VaCuNm0Rj{N6aWAK2mk;8Apo`$"
        "?x@!R000&Q001fg0000000000004jito8r^b8=%ZVrgb?V_|G(Wo>O?FJ*RNY;|FDX>V>XaB^j2X=8M0Z*FrgaCuNm0Rj{N6aWAK"
        "2mk;8ApjKRY+OJ9002V(001Tc0000000000004ji%lH5Qb8=%ZVrgb?V_|G(Wo>O?FKKROWpZV1V`VR2UukY>bYEXCaCuNm0Rj{N"
        "6aWAK2mk;8Apk0oN6O#_00870001Ze0000000000004jiPx$}<b8=%ZVrgb?V_|G(Wo>O?FKKROWpZV1V`VRAWo~71VRUJ4ZZ2?n"
        "P)h*<6aW+e000O8001EXb>f5@DFFZg<N^QyBLDyZ0000000000fB|&@0RVGyV=rQ9W^ZF*Y-eR{ZDB8MZ)9a`b1z?CX>MtBUtcb8"
        "c~DCM0u%rg000080000X063DA*mDN}06rH003iSX00000000000DuAC0s#PXa$_%IX=ZO@VQgn*ZEayMZEs{{Y;!MSa%p5|WiD`e"
        "P)h*<6aW+e000O8001EX32CDWzX1RMLjwQ+AOHXW0000000000fB}&T0RVGyV=rQ9W^ZF*Y-eR{ZDB8MZ)9a`b1!3IV`ybAaCuNm"
        "0Rj{N6aWAK2mk;8Appoma~4$u005;6001rk0000000000004jijtl_+b8=%ZVrgb?V_|G(Wo>O?FKusRWo&aVV`yb#YjAIAZggLC"
        "a$#<BW@U0NaCuNm0Rj{N6aWAK2mk;8AprR&#v5=0004st001ul0000000000004jiFA)I%b8=%ZVrgb?V_|G(Wo>O?FKusRWo&aV"
        "V{c?-UuSY*aA;p}a&Ky7V{~tFE^v8JO928D02BZK00;m803iVC=Og>b0ssId2LJ#i00000000000001h0qhh30CRF<FJftCZ)0I>"
        "XJu_|VJ~fOWMyn~FJ*0FWn^S&ZfA2YaCuNm0Rj{N6aWAK2mk;8Apo=le##O90006A001cf0000000000004ji_ZI;Gb8=%ZVrgb?"
        "V_|G(Wo>O?FKusRWo&aVX>MtBX<=+>dSP^FZ*DGdc~DCM0u%rg000080000X0LG+}9mfU$0E-p?04o3h00000000000Du8d8vy`w"
        "a$_%IX=ZO@VQgn*ZEayMZEs{{Y;!MdWpZa_WM5%)bY*UIX>V>WaCuNm0Rj{N6aWAK2mk;8ApjneqI9tW004sq001BW0000000000"
        "004jiU?Kqkb8=%ZVrgb?V_|G(Wo>O?FKusRWo&aVZEs{{Y%XwlP)h*<6aW+e000O8001EXZAJ{~01p5Fq%!~jApigX0000000000"
        "fB{b>0RVGyV=rQ9W^ZF*Y-eR{ZDB8MZ)9a`b1!aTbZK^FE^v8JO928D02BZK00;m803iUYS>d*S2LJ#~7XSby00000000000001h"
        "0gf~Q0CRF<FJftCZ)0I>XJu_|VJ~fOWMyn~FK}{YbaG*7Ze?UHaCuNm0Rj{N6aWAK2mk;8ApjLeczT-!001Nq001Na0000000000"
        "004jiP&@$ub8=%ZVrgb?V_|G(Wo>O?FKusRWo&aVa%E;^a%FB~WiD`eP)h*<6aW+e000O8001EXgBY%lJq7>(yAS{XA^-pY00000"
        "00000fB`K+0RVGyV=rQ9W^ZF*Y-eR{ZDB8MZ)9a`b1!mlb#!TNXD)DgP)h*<6aW+e000O8001EX#sX1W-V^`;NL2s;Bme*a00000"
        "00000fB~vV0RVGyV=rQ9W^ZF*Y-eR{ZDB8RWpHnEX>@OLd0%sMb!2%iaCuNm0Rj{N6aWAK2mk;8Apo`dk8j=u007w#000~S00000"
        "00000004ji%3T2fb8=%ZVrgb?V_|G(Wo>O?FLPsIW@c|}WG--dP)h*<6aW+e000O8001EX+fWH2jsO4vy#N3JB>(^b0000000000"
        "fC1uT0RVGyV=rQ9W^ZF*Y-eR{ZDB8Ta$#w1X>MmPUtei%X>?y-E^v8JO928D02BZK00;m803iTQ_x$jd1^@s*6aWAy0000000000"
        "0001h0kdQQ0CRF<FJftCZ)0I>XJu_|VJ~!YVQFq@Zf7rJb#h^JWn^D-W^^uac~DCM0u%rg000080000X0EMl(<sJb50ObJy04V?f"
        "00000000000Du9HYXJara$_%IX=ZO@VQgn*ZEayMbaG*7ZfS03FJx(RbZKmCVRUJ4ZZ2?nP)h*<6aW+e000O8001EXY;1#b7!Cjc"
        "vo8PuDF6Tf0000000000fC2Pu0RVGyV=rQ9W^ZF*Y-eR{ZDB8Ta$#w1X>MmPZDnn5a(Q2Ha&KZ~E^v8JO928D02BZK00;m803iSx"
        "W4xPK0RRA$0ssIf00000000000001h0aJSc0CRF<FJftCZ)0I>XJu_|VJ~!YVQFq@Zf7rVVrpe$bZK^Fb1rasP)h*<6aW+e000O8"
        "001EXE#n{Uun+(Mmp=diC;$Ke0000000000fC2D)0RVGyV=rQ9W^ZF*Y-eR{ZDB8Ta$#w1X>MmPZ)Rp}X>Mg-b7pidaCuNm0Rj{N"
        "6aWAK2mk;8Apo>CALbJX003qj001KZ0000000000004ji=8XXWb8=%ZVrgb?V_|G(Wo>O?FLZKYX>Mt5XD@JXY-wY8E^v8JO928D"
        "02BZK00;m803iTufR+o10RRAr0ssIe00000000000001h0Xmie0CRF<FJftCZ)0I>XJu_|VJ~!YVQFq@Zf7rWa&Km7Y-w(1E^v8J"
        "O928D02BZK00;m803iVA%C(=s2mk;t9RL6;00000000000001h0SA}?0CRF<FJftCZ)0I>XJu_|VJ~!YVQFq@Zf7rYWpHnEX>@OL"
        "d0%s8bS`jtP)h*<6aW+e000O8001EX+W}19Bmw{c9|QmZBme*a0000000000fB_Jp0RVGyV=rQ9W^ZF*Y-eR{ZDB8Ta$#w1X>MmP"
        "baG*7Ze?;VaCuNm0Rj{N6aWAK2mk;8ApjsPk_V*@007%6000L70000000000004jicB26ROi4pUPE$otO9ci10000=07?M=82|u4"
        "vH<`900"
    )
    bundle_sha = "dc73a1f3b5f73e2423374a34a2eeefe113dbd5936f5f226cae5f389a7d2c7fab"
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
