import streamlit as st
import pickle
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder

st.set_page_config(page_title="MobiSense AI",page_icon="📱",layout="wide",initial_sidebar_state="collapsed")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0;}
html,body,[data-testid="stAppViewContainer"]{background:#09090F !important;color:#e2e8f0 !important;font-family:'Inter',sans-serif !important;}
[data-testid="stAppViewContainer"]{background:radial-gradient(ellipse 80% 50% at 15% 0%,rgba(0,229,255,0.08) 0%,transparent 55%),radial-gradient(ellipse 60% 45% at 85% 100%,rgba(124,58,237,0.09) 0%,transparent 55%),radial-gradient(ellipse 40% 30% at 50% 40%,rgba(255,78,205,0.04) 0%,transparent 50%),#09090F !important;}
[data-testid="stHeader"],[data-testid="stToolbar"]{display:none !important;}
.block-container{padding:0 2.2rem 6rem 2.2rem !important;max-width:1380px !important;}

.hero{padding:3.5rem 0 2rem;border-bottom:1px solid rgba(255,255,255,0.06);margin-bottom:2rem;}
.hero-pill{font-family:'JetBrains Mono',monospace;font-size:0.58rem;font-weight:500;letter-spacing:0.2em;text-transform:uppercase;color:#00E5FF;background:rgba(0,229,255,0.07);border:1px solid rgba(0,229,255,0.2);padding:0.32rem 0.9rem;border-radius:20px;display:inline-block;margin-bottom:1.5rem;}
.hero-title{font-size:clamp(2.8rem,5.5vw,4.5rem);font-weight:800;letter-spacing:-0.045em;line-height:1.03;color:#f8fafc;margin-bottom:0.8rem;}
.hero-title .gx{background:linear-gradient(135deg,#00E5FF 0%,#7C3AED 50%,#FF4ECD 100%);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;}
.hero-sub{font-size:0.9rem;color:#64748b;line-height:1.7;max-width:440px;margin-bottom:2rem;}

@keyframes neon-pulse{0%{background-position:0% 50%}50%{background-position:100% 50%}100%{background-position:0% 50%}}
.stat-wrap{padding:1.5px;border-radius:12px;margin-bottom:2.5rem;background:linear-gradient(90deg,#00E5FF,#7C3AED,#FF4ECD,#00E5FF);background-size:300%;animation:neon-pulse 4s ease infinite;}
.stat-inner{display:flex;border-radius:11px;background:#0f1019;overflow:hidden;}
.si{flex:1;padding:1.2rem 1.5rem;border-right:1px solid rgba(255,255,255,0.05);}
.si:last-child{border-right:none;}
.sv{font-family:'JetBrains Mono',monospace;font-size:1.9rem;font-weight:600;color:#00E5FF;line-height:1;}
.sl{font-size:0.55rem;font-weight:600;text-transform:uppercase;letter-spacing:0.14em;color:#334155;margin-top:0.45rem;}

[data-testid="stTabs"]>div:first-child{background:transparent !important;border-bottom:1px solid rgba(255,255,255,0.06) !important;gap:0 !important;padding:0 !important;}
button[data-baseweb="tab"]{background:transparent !important;color:#475569 !important;font-family:'Inter',sans-serif !important;font-size:0.88rem !important;font-weight:600 !important;padding:0.85rem 1.4rem !important;border-bottom:2px solid transparent !important;}
button[data-baseweb="tab"]:hover{color:#00E5FF !important;}
button[aria-selected="true"][data-baseweb="tab"]{color:#00E5FF !important;border-bottom:2px solid #00E5FF !important;background:transparent !important;}

.eye{font-family:'JetBrains Mono',monospace;font-size:0.56rem;font-weight:500;letter-spacing:0.2em;text-transform:uppercase;color:#00E5FF;margin-bottom:0.4rem;}
.stl{font-size:1.4rem;font-weight:700;color:#f8fafc;letter-spacing:-0.025em;margin-bottom:1.4rem;}

.cgrid{display:grid;grid-template-columns:1fr 1fr;gap:1rem;margin:1rem 0 1.4rem;}
.cc{border-radius:12px;padding:1.5rem;border:1px solid;position:relative;overflow:hidden;}
.cc::before{content:'';position:absolute;top:0;left:0;right:0;height:3px;}
.cc0{background:rgba(239,68,68,0.04);border-color:rgba(239,68,68,0.14);}
.cc0::before{background:linear-gradient(90deg,#ef4444,#FF4ECD);}
.cc1{background:rgba(0,229,255,0.04);border-color:rgba(0,229,255,0.14);}
.cc1::before{background:linear-gradient(90deg,#00E5FF,#7C3AED);}
.c-i{font-size:1.4rem;margin-bottom:0.7rem;}
.c-n{font-weight:700;font-size:0.88rem;color:#f8fafc;margin-bottom:0.5rem;}
.c-s{font-size:0.78rem;color:#475569;line-height:1.8;}
.c-s strong{color:#94a3b8;}

.ins{background:rgba(0,229,255,0.03);border:1px solid rgba(0,229,255,0.1);border-left:3px solid #00E5FF;border-radius:8px;padding:1.2rem 1.5rem;margin:1.2rem 0;}
.ins h4{font-size:0.85rem;font-weight:700;color:#00E5FF;margin-bottom:0.4rem;}
.ins p{font-size:0.85rem;color:#64748b;line-height:1.75;}

/* MATCHING PHONE CARD */
.mcard{background:rgba(255,255,255,0.02);border:1px solid rgba(255,255,255,0.06);border-radius:10px;padding:1rem 1.3rem;display:flex;align-items:center;gap:1rem;margin-bottom:0.45rem;transition:all 0.2s;}
.mcard:hover{border-color:rgba(0,229,255,0.3);background:rgba(0,229,255,0.03);}
.mc-icon{font-size:1.5rem;min-width:2rem;text-align:center;}
.mc-info{flex:1;}
.mc-model{font-weight:700;font-size:0.97rem;color:#f8fafc;}
.mc-brand{font-size:0.72rem;color:#475569;margin-top:2px;}
.mc-meta{text-align:center;min-width:70px;}
.mc-val{font-family:'JetBrains Mono',monospace;font-size:0.9rem;font-weight:500;color:#94a3b8;}
.mc-lbl{font-size:0.52rem;color:#334155;text-transform:uppercase;letter-spacing:0.1em;margin-top:2px;}

/* SELECTED PHONE CARD */
.scard{background:rgba(0,229,255,0.05);border:1.5px solid rgba(0,229,255,0.4);border-radius:10px;padding:1.1rem 1.4rem;display:flex;align-items:center;gap:1rem;margin-bottom:1rem;box-shadow:0 0 24px rgba(0,229,255,0.07);}
.sc-icon{font-size:1.6rem;min-width:2.2rem;text-align:center;}
.sc-info{flex:1;}
.sc-model{font-weight:700;font-size:1.05rem;color:#f8fafc;}
.sc-brand{font-size:0.72rem;color:#0891b2;margin-top:2px;font-weight:600;}
.sc-meta{text-align:center;min-width:72px;}
.sc-val{font-family:'JetBrains Mono',monospace;font-size:0.95rem;font-weight:600;color:#00E5FF;}
.sc-lbl{font-size:0.52rem;color:#334155;text-transform:uppercase;letter-spacing:0.1em;margin-top:2px;}

/* REC CARD */
.rc{background:rgba(255,255,255,0.015);border:1px solid rgba(255,255,255,0.06);border-radius:10px;padding:1rem 1.3rem;display:flex;align-items:center;gap:1rem;margin-bottom:0.45rem;}
.rc:hover{border-color:rgba(0,229,255,0.25);background:rgba(0,229,255,0.025);}
.rk{font-family:'JetBrains Mono',monospace;font-size:0.68rem;color:#334155;min-width:1.8rem;font-weight:600;}
.ri{font-size:1.3rem;}
.rf{flex:1;}
.rn{font-weight:700;font-size:0.95rem;color:#f8fafc;}
.rb{font-size:0.72rem;color:#475569;margin-top:2px;}
.rm{text-align:center;min-width:62px;}
.rv{font-family:'JetBrains Mono',monospace;font-size:0.86rem;font-weight:500;color:#94a3b8;}
.rl{font-size:0.52rem;color:#334155;text-transform:uppercase;letter-spacing:0.1em;margin-top:2px;}
.rsv{font-family:'JetBrains Mono',monospace;font-size:1rem;font-weight:700;color:#00E5FF;}
.rsb{height:3px;border-radius:2px;margin-top:4px;background:linear-gradient(90deg,#00E5FF,#7C3AED);}
.rsl{font-size:0.5rem;color:#334155;margin-top:3px;}

/* BADGES */
.bg-hi{font-family:'JetBrains Mono',monospace;font-size:0.7rem;font-weight:600;padding:0.28rem 0.7rem;border-radius:20px;background:rgba(0,229,255,0.1);border:1px solid rgba(0,229,255,0.25);color:#00E5FF;}
.bg-md{font-family:'JetBrains Mono',monospace;font-size:0.7rem;font-weight:600;padding:0.28rem 0.7rem;border-radius:20px;background:rgba(124,58,237,0.1);border:1px solid rgba(124,58,237,0.25);color:#a78bfa;}
.bg-lo{font-family:'JetBrains Mono',monospace;font-size:0.7rem;font-weight:600;padding:0.28rem 0.7rem;border-radius:20px;background:rgba(255,78,205,0.1);border:1px solid rgba(255,78,205,0.25);color:#FF4ECD;}
.sat{font-size:0.68rem;font-weight:600;padding:0.26rem 0.65rem;border-radius:20px;background:rgba(0,229,255,0.1);border:1px solid rgba(0,229,255,0.22);color:#00E5FF;}
.dis{font-size:0.68rem;font-weight:600;padding:0.26rem 0.65rem;border-radius:20px;background:rgba(239,68,68,0.1);border:1px solid rgba(239,68,68,0.22);color:#f87171;}

/* FEATURE GRID */
.fg{display:grid;grid-template-columns:repeat(3,1fr);gap:0.55rem;margin:1rem 0;}
.fi{background:rgba(255,255,255,0.02);border:1px solid rgba(255,255,255,0.05);border-radius:8px;padding:0.85rem 1rem;}
.fi:hover{border-color:rgba(0,229,255,0.2);}
.fl{font-size:0.57rem;color:#475569;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:0.35rem;}
.fv{font-family:'JetBrains Mono',monospace;font-size:0.97rem;font-weight:500;color:#e2e8f0;}
.fb{height:2px;border-radius:1px;margin-top:7px;background:linear-gradient(90deg,#00E5FF,#7C3AED);}

/* FILTER & STEP PANELS */
.panel{background:rgba(0,229,255,0.025);border:1px solid rgba(0,229,255,0.1);border-radius:10px;padding:1.2rem 1.5rem;margin-bottom:1.2rem;}
.step-num{font-family:'JetBrains Mono',monospace;font-size:0.56rem;font-weight:600;letter-spacing:0.2em;text-transform:uppercase;color:#00E5FF;margin-bottom:0.3rem;}
.step-desc{font-size:0.8rem;color:#64748b;margin-top:0.15rem;}

/* DIVIDER */
.div{border:none;border-top:1px solid rgba(255,255,255,0.05);margin:1.5rem 0;}

[data-testid="stSelectbox"]>div>div{background:rgba(255,255,255,0.03) !important;border:1px solid rgba(255,255,255,0.08) !important;border-radius:8px !important;color:#e2e8f0 !important;}
[data-testid="stMetricValue"]{font-family:'JetBrains Mono',monospace !important;color:#f8fafc !important;font-size:1.45rem !important;}
[data-testid="stMetricLabel"]{color:#334155 !important;font-size:0.55rem !important;text-transform:uppercase !important;letter-spacing:0.12em !important;}
h1,h2,h3{color:#f8fafc !important;font-family:'Inter',sans-serif !important;}
::-webkit-scrollbar{width:3px;}
::-webkit-scrollbar-track{background:#09090F;}
::-webkit-scrollbar-thumb{background:rgba(0,229,255,0.3);border-radius:2px;}
</style>
""", unsafe_allow_html=True)

# ── LOAD ──────────────────────────────────────────────────────
@st.cache_data
def load_all():
    km  = pickle.load(open('model_files/kmeans_model.pkl','rb'))
    sim = pickle.load(open('model_files/similarity_matrix.pkl','rb'))
    pdf = pd.read_csv('model_files/product_data.csv')
    rdf = pd.read_csv('model_files/mobile_clustered.csv')
    return km, sim, pdf, rdf

kmeans, similarity_matrix, product_df, review_df = load_all()
MODEL_IDX = {r['model']: i for i, r in product_df.iterrows()}
ICONS = {'Apple':'🍎','Google':'🔵','Samsung':'🌀','Xiaomi':'⚡','Realme':'💎','Motorola':'📡','OnePlus':'🔴'}
PAL   = ['#00E5FF','#7C3AED','#FF4ECD','#10b981','#eab308','#ef4444','#f97316']
PC    = 'usd_price' if 'usd_price' in product_df.columns else 'avg_price'

def get_price(m):
    """Return avg_price from product_df — the mean USD-equivalent price across all reviews."""
    row = product_df[product_df['model']==m]
    return int(round(row['avg_price'].values[0])) if len(row)>0 else 0

def mkl(**kw):
    d = dict(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Inter',color='#475569',size=11),
        title_font=dict(family='Inter',color='#f1f5f9',size=13),
        legend=dict(bgcolor='rgba(0,0,0,0)',font=dict(color='#64748b',size=10)),
        margin=dict(t=44,b=22,l=12,r=12),
        xaxis=dict(gridcolor='rgba(255,255,255,0.04)',linecolor='rgba(255,255,255,0.04)',tickfont=dict(color='#475569',size=10)),
        yaxis=dict(gridcolor='rgba(255,255,255,0.04)',linecolor='rgba(255,255,255,0.04)',tickfont=dict(color='#475569',size=10)),
        hoverlabel=dict(bgcolor='#0f1019',font=dict(color='#f8fafc',size=12),bordercolor='rgba(0,229,255,0.3)'))
    d.update(kw)
    return d

def sax(t): return dict(backgroundcolor='rgba(0,0,0,0)',gridcolor='rgba(255,255,255,0.05)',showbackground=True,title=t,tickfont=dict(color='#334155',size=8))

def sbadge(s):
    p=int(s*100)
    if p>=65: return f'<span class="bg-hi">★ {p}% match</span>'
    elif p>=35: return f'<span class="bg-md">★ {p}% match</span>'
    return f'<span class="bg-lo">★ {p}% match</span>'

def cbadge(dc,pct):
    return f'<span class="sat">✅ Satisfied ({pct:.0f}%)</span>' if dc==1 else f'<span class="dis">⚠️ Dissatisfied ({pct:.0f}%)</span>'

def bw(v,mx): return max(int((v/mx)*100),2)

# ── HERO ──────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="hero-pill">📱 K-Means · Cosine Similarity · Partial Hybrid</div>
  <div class="hero-title">Mobile Review<br><span class="gx">Intelligence Engine</span></div>
  <div class="hero-sub">47,550 reviews · 7 brands · 8 countries · 22 models — cluster analysis and smart phone recommendations.</div>
  <div class="stat-wrap"><div class="stat-inner">
    <div class="si"><div class="sv">47.5K</div><div class="sl">Reviews</div></div>
    <div class="si"><div class="sv">22</div><div class="sl">Models</div></div>
    <div class="si"><div class="sv">K=2</div><div class="sl">Clusters</div></div>
    <div class="si"><div class="sv">7</div><div class="sl">Brands</div></div>
    <div class="si"><div class="sv">8</div><div class="sl">Countries</div></div>
  </div></div>
</div>
""", unsafe_allow_html=True)

t1,t2,t3 = st.tabs(["📊  EDA & Overview","🗂️  Cluster Analysis","🔍  Recommendations"])

# ════════════════════════════════════════════════════════════
# TAB 1 — EDA
# ════════════════════════════════════════════════════════════
with t1:
    st.markdown('<div class="eye">Data Overview</div><div class="stl">Exploratory Analysis</div>',unsafe_allow_html=True)
    m1,m2,m3,m4=st.columns(4)
    m1.metric("Total Reviews","47,550")
    m2.metric("Avg Price",f"${review_df['price_usd'].mean():.0f}")
    m3.metric("Avg Rating",f"{review_df['rating'].mean():.2f} ⭐")
    m4.metric("Verified",f"{review_df['verified_purchase'].mean()*100:.0f}%")
    st.markdown("<br>",unsafe_allow_html=True)

    bc=review_df['brand'].value_counts().sort_values()
    fig1=go.Figure(go.Bar(x=bc.values,y=bc.index,orientation='h',
        marker=dict(color=list(range(len(bc))),colorscale=[[0,'#0a2540'],[0.5,'#0891b2'],[1,'#00E5FF']],line=dict(color='rgba(0,229,255,0.2)',width=1)),
        text=[f'{v:,}' for v in bc.values],textposition='outside',
        textfont=dict(color='#00E5FF',size=11,family='JetBrains Mono'),
        hovertemplate='<b>%{y}</b><br>%{x:,} reviews<extra></extra>'))
    fig1.update_layout(title='Reviews by Brand',height=300,**mkl())
    st.plotly_chart(fig1,use_container_width=True)

    ca,cb=st.columns(2)
    with ca:
        sent=review_df['sentiment'].value_counts()
        cpie=[{'Positive':'#00E5FF','Neutral':'#7C3AED','Negative':'#ef4444'}.get(s,'#64748b') for s in sent.index]
        fig2=go.Figure(go.Pie(labels=sent.index,values=sent.values,hole=0.62,
            marker=dict(colors=cpie,line=dict(color='#09090F',width=3)),textfont=dict(color='#64748b',size=10),
            hovertemplate='%{label}<br>%{value:,} · %{percent}<extra></extra>'))
        fig2.add_annotation(text='Sentiment',x=0.5,y=0.5,font=dict(size=10,color='#334155'),showarrow=False)
        fig2.update_layout(title='Sentiment Distribution',height=300,showlegend=True,
            legend=dict(orientation='h',y=-0.08,bgcolor='rgba(0,0,0,0)',font=dict(color='#64748b',size=10)),
            paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',
            font=dict(family='Inter',color='#475569'),title_font=dict(family='Inter',color='#f1f5f9',size=13),
            margin=dict(t=44,b=20,l=12,r=12),hoverlabel=dict(bgcolor='#0f1019',font=dict(color='#f8fafc',size=12)))
        st.plotly_chart(fig2,use_container_width=True)
    with cb:
        pb=review_df.groupby('brand')['price_usd'].mean().sort_values(ascending=False).reset_index()
        fig3=go.Figure(go.Bar(x=pb['brand'],y=pb['price_usd'],
            marker=dict(color=pb['price_usd'],colorscale=[[0,'#0a2540'],[0.5,'#0891b2'],[1,'#00E5FF']],line=dict(color='rgba(0,229,255,0.2)',width=1)),
            text=[f'${v:.0f}' for v in pb['price_usd']],textposition='outside',textfont=dict(color='#64748b',size=10),
            hovertemplate='<b>%{x}</b><br>$%{y:.0f}<extra></extra>'))
        fig3.update_layout(title='Avg Price by Brand',height=300,**mkl())
        st.plotly_chart(fig3,use_container_width=True)

    rc=review_df['rating'].value_counts().sort_index().reset_index(); rc.columns=['rating','count']
    fig4=go.Figure(go.Scatter(x=rc['rating'],y=rc['count'],fill='tozeroy',fillcolor='rgba(0,229,255,0.06)',
        line=dict(color='#00E5FF',width=2),mode='lines+markers',
        marker=dict(color='#00E5FF',size=8,line=dict(color='#0891b2',width=2)),
        hovertemplate='Rating %{x}⭐ — %{y:,}<extra></extra>'))
    fig4.update_layout(title='Rating Distribution',height=240,
        xaxis=dict(tickvals=[1,2,3,4,5],ticktext=['1⭐','2⭐','3⭐','4⭐','5⭐'],gridcolor='rgba(255,255,255,0.04)',linecolor='rgba(255,255,255,0.04)',tickfont=dict(color='#475569',size=10)),
        yaxis=dict(gridcolor='rgba(255,255,255,0.04)',linecolor='rgba(255,255,255,0.04)',tickfont=dict(color='#475569',size=10)),
        paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Inter',color='#475569'),title_font=dict(family='Inter',color='#f1f5f9',size=13),
        margin=dict(t=44,b=22,l=12,r=12),hoverlabel=dict(bgcolor='#0f1019',font=dict(color='#f8fafc',size=12)))
    st.plotly_chart(fig4,use_container_width=True)

    rb=review_df.groupby('brand')['rating'].mean().sort_values().reset_index()
    fig5=go.Figure()
    for _,row in rb.iterrows():
        fig5.add_trace(go.Scatter(x=[0,row['rating']],y=[row['brand'],row['brand']],mode='lines',line=dict(color='rgba(0,229,255,0.08)',width=2),showlegend=False))
    fig5.add_trace(go.Scatter(x=rb['rating'],y=rb['brand'],mode='markers+text',
        marker=dict(color=PAL[:len(rb)],size=13,line=dict(color='#09090F',width=2)),
        text=[f'{v:.2f}' for v in rb['rating']],textposition='middle right',
        textfont=dict(color='#64748b',size=10,family='JetBrains Mono'),showlegend=False))
    fig5.update_layout(title='Avg Rating by Brand',height=270,
        **mkl(xaxis=dict(range=[0,5.6],gridcolor='rgba(255,255,255,0.04)',linecolor='rgba(255,255,255,0.04)',tickfont=dict(color='#475569',size=10))))
    st.plotly_chart(fig5,use_container_width=True)

    fig6=go.Figure(go.Histogram(x=review_df['age'],nbinsx=25,
        marker=dict(color='#0891b2',opacity=0.85,line=dict(color='rgba(0,229,255,0.25)',width=1)),
        hovertemplate='Age %{x}<br>%{y:,}<extra></extra>'))
    fig6.update_layout(title='Reviewer Age Distribution',height=240,**mkl())
    st.plotly_chart(fig6,use_container_width=True)

# ════════════════════════════════════════════════════════════
# TAB 2 — CLUSTER ANALYSIS
# ════════════════════════════════════════════════════════════
with t2:
    st.markdown('<div class="eye">K-Means · K=2</div><div class="stl">Two Customer Segments</div>',unsafe_allow_html=True)
    st.markdown("""
    <div class="cgrid">
      <div class="cc cc0"><div class="c-i">⚠️</div><div class="c-n">Cluster 0 — Dissatisfied</div>
        <div class="c-s"><strong>22,017</strong> reviews · Avg Price <strong>$689</strong><br>Avg Rating <strong>2.07 ⭐</strong><br><span style="color:#f87171;font-size:0.68rem;">High negative &amp; neutral sentiment</span></div></div>
      <div class="cc cc1"><div class="c-i">✅</div><div class="c-n">Cluster 1 — Satisfied</div>
        <div class="c-s"><strong>25,533</strong> reviews · Avg Price <strong>$690</strong><br>Avg Rating <strong>4.01 ⭐</strong><br><span style="color:#00E5FF;font-size:0.68rem;">Predominantly positive sentiment</span></div></div>
    </div>
    <div class="ins"><h4>💡 The Price Paradox</h4><p>Both clusters average nearly identical prices ($689 vs $690) yet satisfaction diverges dramatically — 2.07⭐ vs 4.01⭐. Price alone cannot predict happiness. Product quality is the real differentiator.</p></div>
    """,unsafe_allow_html=True)

    st.markdown('<div class="eye" style="margin-top:2rem;">3D PCA Visualization</div>',unsafe_allow_html=True)
    le_t=LabelEncoder(); tmp=review_df.copy()
    for c in ['sentiment','verified_purchase','brand','model','country']:
        tmp[c+'_e']=le_t.fit_transform(tmp[c].astype(str))
    tmp['pl']=np.log1p(tmp['price_usd'])
    ft=['age','pl','rating','battery_life_rating','camera_rating','performance_rating','design_rating','display_rating','helpful_votes','sentiment_e','verified_purchase_e','brand_e','model_e','country_e']
    s3=StandardScaler().fit_transform(tmp[ft]); p3=PCA(n_components=3,random_state=42); r3=p3.fit_transform(s3)
    tmp['p1'],tmp['p2'],tmp['p3']=r3[:,0],r3[:,1],r3[:,2]; vp=sum(p3.explained_variance_ratio_)*100
    fp=go.Figure()
    for c,col,nm in [(0,'#ef4444','Cluster 0'),(1,'#00E5FF','Cluster 1')]:
        sub=tmp[tmp['cluster']==c].sample(min(3000,int((tmp['cluster']==c).sum())),random_state=42)
        fp.add_trace(go.Scatter3d(x=sub['p1'],y=sub['p2'],z=sub['p3'],mode='markers',name=nm,
            marker=dict(color=col,size=2,opacity=0.45,line=dict(width=0))))
    fp.update_layout(title=f'3D PCA — {vp:.0f}% variance',height=480,
        scene=dict(bgcolor='rgba(0,0,0,0)',xaxis=sax('PC1'),yaxis=sax('PC2'),zaxis=sax('PC3')),
        paper_bgcolor='rgba(0,0,0,0)',font=dict(family='Inter',color='#475569'),
        title_font=dict(family='Inter',color='#f1f5f9',size=13),
        legend=dict(bgcolor='rgba(0,0,0,0)',font=dict(color='#64748b',size=11)),
        margin=dict(t=50,b=10,l=10,r=10),hoverlabel=dict(bgcolor='#0f1019',font=dict(color='#f8fafc',size=12)))
    st.plotly_chart(fp,use_container_width=True)

    z1,z2=st.columns(2)
    with z1:
        f7=go.Figure()
        for c,col,nm in [(0,'#ef4444','Cluster 0'),(1,'#00E5FF','Cluster 1')]:
            fc='rgba(239,68,68,0.07)' if c==0 else 'rgba(0,229,255,0.07)'
            f7.add_trace(go.Box(y=review_df[review_df['cluster']==c]['rating'],name=nm,marker_color=col,fillcolor=fc,boxpoints=False,line_color=col))
        f7.update_layout(title='Rating by Cluster',height=300,showlegend=False,**mkl())
        st.plotly_chart(f7,use_container_width=True)
    with z2:
        sc={'Positive':'#00E5FF','Neutral':'#7C3AED','Negative':'#ef4444'}
        f8=go.Figure()
        for c,cn in [(0,'Cluster 0'),(1,'Cluster 1')]:
            sv=review_df[review_df['cluster']==c]['sentiment'].value_counts()
            for s,cnt in sv.items():
                f8.add_trace(go.Bar(name=f'{cn}·{s}',x=[cn],y=[cnt],marker_color=sc.get(s,'#64748b'),opacity=0.88))
        f8.update_layout(title='Sentiment by Cluster',barmode='stack',height=300,**mkl())
        st.plotly_chart(f8,use_container_width=True)

    sr=['battery_life_rating','camera_rating','performance_rating','design_rating','display_rating']
    f9=go.Figure()
    f9.add_trace(go.Bar(x=['Battery','Camera','Perf','Design','Display'],y=review_df[review_df['cluster']==0][sr].mean().values,name='Cluster 0',marker_color='#ef4444',opacity=0.82))
    f9.add_trace(go.Bar(x=['Battery','Camera','Perf','Design','Display'],y=review_df[review_df['cluster']==1][sr].mean().values,name='Cluster 1',marker_color='#00E5FF',opacity=0.82))
    f9.update_layout(title='Sub-Rating Comparison',barmode='group',height=280,
        **mkl(yaxis=dict(range=[0,5],gridcolor='rgba(255,255,255,0.04)',linecolor='rgba(255,255,255,0.04)',tickfont=dict(color='#475569',size=10))))
    st.plotly_chart(f9,use_container_width=True)

    kv=[1,2,3,4,5,6,7,8,9,10]; ine=[665700,485202,451029,424945,405010,386972,375047,362564,355450,344309]
    skk=[2,3,4,5,6,7,8]; svv=[0.2338,0.1337,0.1302,0.1069,0.1125,0.1069,0.1074]
    f10=go.Figure()
    f10.add_trace(go.Scatter(x=kv,y=ine,name='Inertia',line=dict(color='#00E5FF',width=2.5),mode='lines+markers',marker=dict(color='#00E5FF',size=7,line=dict(color='#09090F',width=2)),yaxis='y'))
    f10.add_trace(go.Scatter(x=skk,y=svv,name='Silhouette',line=dict(color='#7C3AED',width=2.5,dash='dot'),mode='lines+markers',marker=dict(color='#7C3AED',size=7,line=dict(color='#09090F',width=2)),yaxis='y2'))
    f10.add_vline(x=2,line_dash='dash',line_color='rgba(0,229,255,0.4)',line_width=1.5,annotation_text='K=2',annotation_font_color='#00E5FF',annotation_font_size=10)
    f10.update_layout(title='Elbow & Silhouette',height=290,
        yaxis=dict(title='Inertia',gridcolor='rgba(255,255,255,0.04)',linecolor='rgba(255,255,255,0.04)',tickfont=dict(color='#475569',size=9),title_font=dict(color='#475569',size=10)),
        yaxis2=dict(title='Silhouette',overlaying='y',side='right',range=[0,0.32],tickfont=dict(color='#475569',size=9),title_font=dict(color='#475569',size=10)),
        xaxis=dict(gridcolor='rgba(255,255,255,0.04)',linecolor='rgba(255,255,255,0.04)',tickfont=dict(color='#475569',size=10)),
        legend=dict(bgcolor='rgba(0,0,0,0)',font=dict(color='#64748b',size=10)),
        paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Inter',color='#475569'),title_font=dict(family='Inter',color='#f1f5f9',size=13),
        margin=dict(t=44,b=22,l=12,r=72),hoverlabel=dict(bgcolor='#0f1019',font=dict(color='#f8fafc',size=12)))
    st.plotly_chart(f10,use_container_width=True)

# ════════════════════════════════════════════════════════════
# TAB 3 — RECOMMENDATIONS
# ════════════════════════════════════════════════════════════
with t3:
    st.markdown('<div class="eye">Partial Hybrid · Cosine Similarity</div><div class="stl">Find Similar Phones</div>',unsafe_allow_html=True)

    # ── STEP 1: FILTERS ──────────────────────────────────────
    st.markdown('<div class="panel"><div class="step-num">Step 1 — Filter</div><div class="step-desc">Narrow down phones by brand, budget, or rating</div></div>',unsafe_allow_html=True)

    fc1,fc2,fc3=st.columns(3)
    with fc1:
        sel_brand=st.selectbox('📱 Brand',['All Brands']+sorted(product_df['brand'].unique().tolist()))
    with fc2:
        sel_price=st.selectbox('💰 Budget (USD price)',['All Prices','Under $500','$500 – $700','$700 – $900','Above $900'])
    with fc3:
        sel_rating=st.selectbox('⭐ Min Rating',['All Ratings','1★ and above','2★ and above','3★ and above','4★ and above','5★'])

    pm={'All Prices':(0,9999),'Under $500':(0,500),'$500 – $700':(500,700),'$700 – $900':(700,900),'Above $900':(900,9999)}
    pmin,pmax=pm[sel_price]
    flt=product_df.copy()
    if sel_brand!='All Brands': flt=flt[flt['brand']==sel_brand]

    # Price filter uses real retail prices
    flt=flt[(flt['avg_price']>=pmin)&(flt['avg_price']<pmax)]

    # Rating filter — 1★/2★/3★ use avg_rating; 4★/5★ use % of individual reviews
    def _pct_rating(m, min_r):
        pr=review_df[review_df['model']==m]
        return (pr['rating']>=min_r).mean()*100 if len(pr)>0 else 0
    if sel_rating=='1★ and above':
        flt=flt[flt['avg_rating']>=1.0]
    elif sel_rating=='2★ and above':
        flt=flt[flt['avg_rating']>=2.0]
    elif sel_rating=='3★ and above':
        flt=flt[flt['avg_rating']>=3.0]
    elif sel_rating=='4★ and above':
        flt=flt[flt['model'].apply(lambda m: _pct_rating(m,4))>=40]
    elif sel_rating=='5★':
        flt=flt[flt['model'].apply(lambda m: _pct_rating(m,5))>=12]
    flt=flt.reset_index(drop=True)

    # Match count pill
    nc='#00E5FF' if len(flt)>0 else '#ef4444'
    st.markdown(f'<div style="margin:0.7rem 0 1rem;"><span style="font-family:\'JetBrains Mono\',monospace;font-size:0.7rem;color:{nc};background:rgba(0,229,255,0.06);border:1px solid rgba(0,229,255,0.15);padding:0.3rem 0.75rem;border-radius:20px;">{len(flt)} phone(s) match your filters</span></div>',unsafe_allow_html=True)

    if len(flt)==0:
        st.markdown('<div class="ins"><h4>No phones match</h4><p>Try a wider budget, different brand, or remove the rating filter. All models average 3.07–3.15⭐ so 4★+ returns nothing.</p></div>',unsafe_allow_html=True)
        st.stop()

    # ── MATCHING PHONES LIST (ALL that match) ────────────────
    st.markdown('<hr class="div"><div class="eye">Matching Phones</div><div style="font-size:0.8rem;color:#64748b;margin-bottom:0.9rem;">All phones that match your filters</div>',unsafe_allow_html=True)

    for _,row in flt.iterrows():
        icon = ICONS.get(row['brand'],'📱')
        ph   = review_df[review_df['model']==row['model']]
        dc   = int(ph['cluster'].value_counts().idxmax())
        pct  = ph['cluster'].value_counts()[dc]/len(ph)*100
        bdg  = cbadge(dc,pct)
        price_show = get_price(row['model'])
        pos_rate = (ph['sentiment']=='Positive').mean()*100
        st.markdown(f"""
        <div class="mcard">
          <div class="mc-icon">{icon}</div>
          <div class="mc-info">
            <div class="mc-model">{row['model']}</div>
            <div class="mc-brand">{row['brand']}</div>
          </div>
          <div class="mc-meta">
            <div class="mc-val">${price_show}</div>
            <div class="mc-lbl">Price (USD)</div>
          </div>
          <div class="mc-meta" style="margin-left:0.8rem;">
            <div class="mc-val">{row['avg_rating']:.2f}⭐</div>
            <div class="mc-lbl">Avg Rating</div>
          </div>
          <div class="mc-meta" style="margin-left:0.8rem;">
            <div class="mc-val">{int(row['num_reviews']):,}</div>
            <div class="mc-lbl">Reviews</div>
          </div>
          <div class="mc-meta" style="margin-left:0.8rem;">
            <div class="mc-val">{pos_rate:.0f}%</div>
            <div class="mc-lbl">Positive</div>
          </div>
          <div style="margin-left:auto;">{bdg}</div>
        </div>
        """, unsafe_allow_html=True)

    # ── STEP 2: SELECT ONE PHONE ─────────────────────────────
    st.markdown('<hr class="div"><div class="panel"><div class="step-num">Step 2 — Choose a Phone</div><div class="step-desc">Pick one phone from the matches above to get recommendations</div></div>',unsafe_allow_html=True)

    phone_options=flt['model'].tolist()
    selected_phone=st.selectbox(
        '🔍 Select Phone Model',
        phone_options,
        format_func=lambda m: f"{ICONS.get(product_df[product_df['model']==m]['brand'].values[0],'📱')}  {m}  —  ${get_price(m)}"
    )

    # Selected phone hero card
    sel_row  = product_df[product_df['model']==selected_phone].iloc[0]
    sel_rev  = review_df[review_df['model']==selected_phone]
    sel_dc   = int(sel_rev['cluster'].value_counts().idxmax())
    sel_pct  = sel_rev['cluster'].value_counts()[sel_dc]/len(sel_rev)*100
    sel_badge= cbadge(sel_dc,sel_pct)
    sel_price= get_price(selected_phone)
    sel_icon = ICONS.get(sel_row['brand'],'📱')
    sel_pos  = (sel_rev['sentiment']=='Positive').mean()*100

    st.markdown(f"""
    <div class="scard">
      <div class="sc-icon">{sel_icon}</div>
      <div class="sc-info">
        <div class="sc-model">{selected_phone}</div>
        <div class="sc-brand">{sel_row['brand']} · Selected</div>
      </div>
      <div class="sc-meta">
        <div class="sc-val">${sel_price}</div>
        <div class="sc-lbl">Price (USD)</div>
      </div>
      <div class="sc-meta" style="margin-left:0.8rem;">
        <div class="sc-val">{sel_row['avg_rating']:.2f}⭐</div>
        <div class="sc-lbl">Rating</div>
      </div>
      <div class="sc-meta" style="margin-left:0.8rem;">
        <div class="sc-val">{int(sel_row['num_reviews']):,}</div>
        <div class="sc-lbl">Reviews</div>
      </div>
      <div class="sc-meta" style="margin-left:0.8rem;">
        <div class="sc-val">{sel_pos:.0f}%</div>
        <div class="sc-lbl">Positive</div>
      </div>
      <div style="margin-left:auto;">{sel_badge}</div>
    </div>
    """, unsafe_allow_html=True)

    # ── STEP 3: RECOMMENDATIONS ──────────────────────────────
    st.markdown('<hr class="div"><div class="eye">Step 3 — Recommended Similar Phones</div>',unsafe_allow_html=True)
    st.markdown(f'<div class="stl">Based on {selected_phone}</div>',unsafe_allow_html=True)
    st.markdown('<div style="font-size:0.8rem;color:#64748b;margin-bottom:1.2rem;">Cosine similarity computed from price · specs · brand features</div>',unsafe_allow_html=True)

    sel_idx=MODEL_IDX[selected_phone]
    scores=sorted(enumerate(similarity_matrix[sel_idx]),key=lambda x:x[1],reverse=True)
    scores=[s for s in scores if s[0]!=sel_idx][:5]

    recs=product_df.iloc[[s[0] for s in scores]][['brand','model',PC,'avg_rating']].copy()
    recs['sim']=[round(s[1],3) for s in scores]
    recs=recs.reset_index(drop=True)

    ms=max(recs['sim'].max(),0.01)
    for i,row in recs.iterrows():
        bww=max(int((max(row['sim'],0)/ms)*100),4)
        bi=ICONS.get(row['brand'],'📱')
        bd=sbadge(row['sim'])
        pr=get_price(row['model'])
        r_rev=review_df[review_df['model']==row['model']]
        r_dc=int(r_rev['cluster'].value_counts().idxmax())
        r_pct=r_rev['cluster'].value_counts()[r_dc]/len(r_rev)*100
        r_cbdg=cbadge(r_dc,r_pct)
        st.markdown(f"""
        <div class="rc">
          <div class="rk">#{i+1}</div>
          <div class="ri">{bi}</div>
          <div class="rf">
            <div class="rn">{row['model']}</div>
            <div class="rb">{row['brand']}</div>
          </div>
          <div class="rm"><div class="rv">${pr}</div><div class="rl">Price</div></div>
          <div class="rm" style="margin-left:0.8rem;"><div class="rv">{row['avg_rating']:.2f}⭐</div><div class="rl">Rating</div></div>
          <div style="margin-left:0.8rem;">{r_cbdg}</div>
          <div style="margin-left:auto;display:flex;flex-direction:column;align-items:flex-end;gap:0.4rem;">
            {bd}
            <div>
              <div class="rsv">{row['sim']:.3f}</div>
              <div class="rsb" style="width:{bww}px;"></div>
              <div class="rsl">cosine score</div>
            </div>
          </div>
        </div>""", unsafe_allow_html=True)

    # ── FEATURE PROFILE ───────────────────────────────────────
    st.markdown('<hr class="div"><div class="eye">Feature Profile</div>',unsafe_allow_html=True)
    st.markdown(f'<div class="stl">{selected_phone}</div>',unsafe_allow_html=True)
    ph=sel_rev
    cl_b=cbadge(sel_dc,sel_pct).replace('class="sat"','class="sat" style="font-size:0.78rem;"').replace('class="dis"','class="dis" style="font-size:0.78rem;"')
    st.markdown(f'<div style="margin-bottom:1rem;">{cl_b}</div>',unsafe_allow_html=True)
    ap=ph['price_usd'].mean(); ar=ph['rating'].mean()
    ab=ph['battery_life_rating'].mean(); ac=ph['camera_rating'].mean()
    apc=ph['performance_rating'].mean(); ad=ph['design_rating'].mean()
    adi=ph['display_rating'].mean(); ah=ph['helpful_votes'].mean()
    aa=ph['age'].mean(); pos=(ph['sentiment']=='Positive').mean()*100
    ver=ph['verified_purchase'].mean()*100
    tc=ph['country'].value_counts().index[0]; ts=ph['source'].value_counts().index[0]

    st.markdown(f"""
    <div class="fg">
      <div class="fi"><div class="fl">Price (USD)</div><div class="fv">${sel_price}</div><div class="fb" style="width:{bw(sel_price,1500)}%;"></div></div>
      <div class="fi"><div class="fl">Avg Rating</div><div class="fv">{ar:.2f} ⭐</div><div class="fb" style="width:{bw(ar,5)}%;"></div></div>
      <div class="fi"><div class="fl">Battery Rating</div><div class="fv">{ab:.2f} / 5</div><div class="fb" style="width:{bw(ab,5)}%;"></div></div>
      <div class="fi"><div class="fl">Camera Rating</div><div class="fv">{ac:.2f} / 5</div><div class="fb" style="width:{bw(ac,5)}%;"></div></div>
      <div class="fi"><div class="fl">Performance</div><div class="fv">{apc:.2f} / 5</div><div class="fb" style="width:{bw(apc,5)}%;"></div></div>
      <div class="fi"><div class="fl">Design Rating</div><div class="fv">{ad:.2f} / 5</div><div class="fb" style="width:{bw(ad,5)}%;"></div></div>
      <div class="fi"><div class="fl">Display Rating</div><div class="fv">{adi:.2f} / 5</div><div class="fb" style="width:{bw(adi,5)}%;"></div></div>
      <div class="fi"><div class="fl">Helpful Votes</div><div class="fv">{ah:.1f}</div><div class="fb" style="width:{bw(ah,9)}%;"></div></div>
      <div class="fi"><div class="fl">Reviewer Age</div><div class="fv">{aa:.0f} yrs</div><div class="fb" style="width:{bw(aa,70)}%;"></div></div>
      <div class="fi"><div class="fl">Positive Reviews</div><div class="fv">{pos:.0f}%</div><div class="fb" style="width:{bw(pos,100)}%;"></div></div>
      <div class="fi"><div class="fl">Verified Purchase</div><div class="fv">{ver:.0f}%</div><div class="fb" style="width:{bw(ver,100)}%;"></div></div>
      <div class="fi"><div class="fl">Top Country · Platform</div><div class="fv" style="font-size:0.82rem;">{tc} · {ts}</div><div class="fb" style="width:55%;"></div></div>
    </div>""", unsafe_allow_html=True)

    # ── SIMILARITY CHART ──────────────────────────────────────
    st.markdown('<hr class="div"><div class="eye">Similarity Score Chart</div>',unsafe_allow_html=True)
    fs=go.Figure(go.Bar(
        x=recs['model'], y=recs['sim'],
        marker=dict(color=recs['sim'],colorscale=[[0,'#0a2540'],[0.4,'#0891b2'],[1,'#00E5FF']],line=dict(color='rgba(0,229,255,0.3)',width=1)),
        text=[f'{v:.3f}' for v in recs['sim']],textposition='outside',
        textfont=dict(color='#00E5FF',size=11,family='JetBrains Mono'),
        hovertemplate='<b>%{x}</b><br>Cosine: %{y:.3f}<extra></extra>'))
    fs.update_layout(title=f'Cosine Similarity vs {selected_phone}',height=280,
        **mkl(yaxis=dict(range=[0,min(ms*1.35,1.05)],gridcolor='rgba(255,255,255,0.04)',linecolor='rgba(255,255,255,0.04)',tickfont=dict(color='#475569',size=10))))
    st.plotly_chart(fs,use_container_width=True)

    st.markdown('<br>',unsafe_allow_html=True)