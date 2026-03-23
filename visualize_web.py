import streamlit as st
import numpy as np
import torch
import smplx
import plotly.graph_objects as go
import time
import os

# --- PAGE SETUP ---
st.set_page_config(page_title="NPZ Motion Visualizer", layout="wide")
st.title("🕺 NPZ 3D Motion Visualizer")
st.markdown("Interactive 3D visualization of SMPL-X motion data.")

# --- SIDEBAR: Load Data ---
st.sidebar.header("Data Configuration")
npz_path = st.sidebar.text_input("NPZ File Path", "output/intro_output.npz")

if not os.path.exists(npz_path):
    st.error(f"File not found: {npz_path}")
    st.stop()

@st.cache_resource
def load_and_compute_data(path):
    data = np.load(path)
    poses = data["poses"]              # (T, 165)
    trans = data["trans"]              # (T, 3)
    expressions = data["expressions"]  # (T, 100)
    
    # SMPL-X usually provides betas in the first frame or shared
    if "betas" in data:
        betas = data["betas"][:10]
    else:
        betas = np.zeros(10)

    T = poses.shape[0]
    
    # Load SMPL-X Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = smplx.create(
        model_path="models",
        model_type="smplx",
        gender="neutral",
        use_pca=False,
        num_expression_coeffs=100,
        batch_size=T
    ).to(device)
    
    # Compute Vertices (Batched for speed)
    poses_t  = torch.tensor(poses, dtype=torch.float32).to(device)
    trans_t  = torch.tensor(trans, dtype=torch.float32).to(device)
    expr_t   = torch.tensor(expressions, dtype=torch.float32).to(device)
    beta_t   = torch.tensor(betas, dtype=torch.float32).unsqueeze(0).expand(T, -1).to(device)
    
    with torch.no_grad():
        output = model(
            betas=beta_t,
            global_orient=poses_t[:, :3],
            body_pose=poses_t[:, 3:66],
            jaw_pose=poses_t[:, 66:69],
            leye_pose=poses_t[:, 69:72],
            reye_pose=poses_t[:, 72:75],
            left_hand_pose=poses_t[:, 75:120],
            right_hand_pose=poses_t[:, 120:165],
            expression=expr_t,
            transl=trans_t
        )
    
    return output.vertices.cpu().numpy(), model.faces, T, trans

# --- CACHED COMPUTATION ---
with st.spinner("Processing motion data and computing 3D mesh..."):
    all_vertices, faces, T, trans = load_and_compute_data(npz_path)

# Precompute global bounds for world-space view
global_min = all_vertices.min(axis=(0, 1))
global_max = all_vertices.max(axis=(0, 1))
global_pad = (global_max - global_min) * 0.05
global_pad = np.where(global_pad == 0, 0.1, global_pad)  # avoid zero padding

# --- ANIMATION CONTROLS ---
col1, col2 = st.columns([1, 3])
mesh_placeholder = None

with col1:
    st.subheader("Controls")
    frame_idx = st.slider("Select Frame", 0, T-1, 0, step=1)
    
    # Stats
    st.info(f"Total Frames: {T}\n\nVertices: {all_vertices.shape[1]}\n\nFaces: {faces.shape[0]}")
    
    view_mode = st.selectbox(
        "View Mode",
        ["Body-centered", "Follow translation", "World-space (fixed axes)"],
        index=0
    )
    performance = st.selectbox(
        "Performance",
        ["Balanced", "Best quality", "Fastest"],
        index=0
    )
    animate_client = st.toggle("Animate in browser", value=True)

with col2:
    # --- PLOTLY 3D MESH ---
    st.subheader(f"3D View - Frame {frame_idx}")
    mesh_placeholder = st.empty()

def build_fig(verts, faces_display, axis_min, axis_max, pad, frames=None, frame_duration=50):
    i = faces_display[:, 0]
    j = faces_display[:, 1]
    k = faces_display[:, 2]
    fig = go.Figure(data=[
        go.Mesh3d(
            x=verts[:, 0],
            y=verts[:, 1],
            z=verts[:, 2],
            i=i, j=j, k=k,
            color='lightgray',
            opacity=1.0,
            flatshading=False,
            lighting=dict(ambient=0.45, diffuse=0.9, specular=0.2, roughness=0.9),
            lightposition=dict(x=6, y=10, z=6)
        )
    ])
    fig.update_layout(
        scene=dict(
            xaxis=dict(nticks=4, range=[axis_min[0] - pad[0], axis_max[0] + pad[0]]),
            yaxis=dict(nticks=4, range=[axis_min[1] - pad[1], axis_max[1] + pad[1]]),
            zaxis=dict(nticks=4, range=[axis_min[2] - pad[2], axis_max[2] + pad[2]]),
            aspectmode='data',
            dragmode='orbit',
            camera=dict(
                up=dict(x=0, y=1, z=0),
                eye=dict(x=1.5, y=1.0, z=1.8),
                projection=dict(type='perspective')
            )
        ),
        margin=dict(r=0, l=0, b=0, t=0),
        height=700
    )
    if frames:
        fig.frames = frames
        fig.update_layout(
            updatemenus=[{
                "type": "buttons",
                "direction": "left",
                "x": 0.0,
                "y": 1.05,
                "showactive": False,
                "buttons": [
                    {
                        "label": "Play",
                        "method": "animate",
                        "args": [None, {
                            "frame": {"duration": frame_duration, "redraw": True},
                            "transition": {"duration": 0},
                            "fromcurrent": True,
                            "mode": "immediate"
                        }]
                    },
                    {
                        "label": "Pause",
                        "method": "animate",
                        "args": [[None], {
                            "frame": {"duration": 0, "redraw": False},
                            "transition": {"duration": 0},
                            "mode": "immediate"
                        }]
                    }
                ]
            }]
        )
    return fig

def get_display_vertices(frame_i):
    verts = all_vertices[frame_i]
    if view_mode == "Body-centered":
        center = verts.mean(axis=0, keepdims=True)
        return verts - center
    if view_mode == "Follow translation":
        return verts - trans[frame_i]
    return verts

if performance == "Best quality":
    face_stride = 1
    frame_stride = 1
elif performance == "Fastest":
    face_stride = 2
    frame_stride = 4
else:
    face_stride = 1
    frame_stride = 2

faces_display = faces[::face_stride]

# Compute stable axis bounds per view mode to avoid flashing/zoom jitter
if "bounds_key" not in st.session_state or st.session_state.bounds_key != npz_path:
    st.session_state.bounds_key = npz_path
    view_bounds = {}
    # Body-centered bounds
    min_bc = np.array([np.inf, np.inf, np.inf], dtype=np.float32)
    max_bc = np.array([-np.inf, -np.inf, -np.inf], dtype=np.float32)
    # Follow-translation bounds
    min_ft = np.array([np.inf, np.inf, np.inf], dtype=np.float32)
    max_ft = np.array([-np.inf, -np.inf, -np.inf], dtype=np.float32)
    for idx in range(T):
        v = all_vertices[idx]
        c = v.mean(axis=0)
        vc = v - c
        min_bc = np.minimum(min_bc, vc.min(axis=0))
        max_bc = np.maximum(max_bc, vc.max(axis=0))
        vt = v - trans[idx]
        min_ft = np.minimum(min_ft, vt.min(axis=0))
        max_ft = np.maximum(max_ft, vt.max(axis=0))
    view_bounds["Body-centered"] = (min_bc, max_bc)
    view_bounds["Follow translation"] = (min_ft, max_ft)
    view_bounds["World-space (fixed axes)"] = (global_min, global_max)
    st.session_state.view_bounds = view_bounds

view_bounds = st.session_state.view_bounds
axis_min, axis_max = view_bounds[view_mode]
pad = (axis_max - axis_min) * 0.08
pad = np.where(pad == 0, 0.1, pad)

if animate_client:
    import math
    indices = list(range(frame_idx, T, frame_stride))
    max_frames = 300
    if len(indices) > max_frames:
        step = math.ceil(len(indices) / max_frames)
        indices = indices[::step]
    frames = []
    for idx in indices:
        v = get_display_vertices(idx)
        frames.append(go.Frame(data=[go.Mesh3d(x=v[:, 0], y=v[:, 1], z=v[:, 2])], name=str(idx)))
    base_verts = get_display_vertices(indices[0] if indices else frame_idx)
    fig = build_fig(base_verts, faces_display, axis_min, axis_max, pad, frames=frames, frame_duration=50)
else:
    base_verts = get_display_vertices(frame_idx)
    fig = build_fig(base_verts, faces_display, axis_min, axis_max, pad)

mesh_placeholder.plotly_chart(fig, use_container_width=True)

st.success("Visualizer Ready!")
