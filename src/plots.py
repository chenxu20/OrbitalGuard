import numpy as np
import plotly.graph_objects as go


def geometry_status(row):
    vals=[row.get('relative_position_'+c,np.nan) for c in 'rtn']
    if not np.isfinite(vals).all():return None,'RTN geometry unavailable: missing position components.'
    norm=float(np.linalg.norm(vals));distance=row.get('miss_distance',np.nan)
    if not np.isfinite(distance):return vals,f'Position norm {norm:,.1f} m; distance unavailable.'
    delta=abs(norm-distance);tolerance=max(10,.02*distance)
    return vals,f'RTN norm {norm:,.1f} m · reported distance {distance:,.1f} m · difference {delta:,.1f} m'+(' — DISCREPANCY exceeds max(10 m, 2%).' if delta>tolerance else ' — within rounding tolerance.')


def geometry_plot(row):
    point,message=geometry_status(row)
    if point is None:return None,message
    fig=go.Figure()
    fig.add_trace(go.Scatter3d(x=[0],y=[0],z=[0],mode='markers',name='Target reference',marker=dict(size=8,color='#7dd3fc')))
    fig.add_trace(go.Scatter3d(x=[point[0]],y=[point[1]],z=[point[2]],mode='markers',name='Relative encounter estimate',marker=dict(size=7,color='#fbbf24')))
    fig.update_layout(template='plotly_dark',paper_bgcolor='#0b1425',height=350,margin=dict(l=0,r=0,t=15,b=0),scene=dict(aspectmode='data',xaxis_title='R (m)',yaxis_title='T (m)',zaxis_title='N (m)'),legend=dict(orientation='h'))
    return fig,message
