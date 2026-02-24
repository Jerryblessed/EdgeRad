import bpy
import os

# --------------------------------------------------
# ⚙️ CONFIGURATION
# --------------------------------------------------
INPUT_FILE = "house2.fbx"  
OUTPUT_FILE = "converted_house2.glb"

# --------------------------------------------------
# 0. GPU SETUP (NVIDIA GTX 1050)
# --------------------------------------------------
def configure_gpu():
    print("⚙️ Configuring GPU acceleration...")
    
    try:
        # 1. Get Cycles Preferences
        prefs = bpy.context.preferences
        cprefs = prefs.addons['cycles'].preferences
        
        # 2. Refresh device list to find the GTX 1050
        cprefs.refresh_devices()
        
        # 3. Set Device Type to CUDA (Best for GTX 10 series)
        # Note: newer cards use 'OPTIX', but 'CUDA' is safe for 1050
        cprefs.compute_device_type = 'CUDA'
        
        # 4. Enable the devices
        found_gpu = False
        for device in cprefs.devices:
            if device.type == 'CUDA':
                device.use = True
                print(f"   ✅ Active Device: {device.name}")
                found_gpu = True
            else:
                device.use = False # Disable CPU if we want pure GPU, or keep True for Hybrid
        
        # 5. Set the Scene to use GPU
        scene = bpy.context.scene
        scene.cycles.device = 'GPU'
        
        if not found_gpu:
            print("   ⚠️ No CUDA GPU found. Falling back to CPU.")
            scene.cycles.device = 'CPU'
            
    except Exception as e:
        print(f"   ⚠️ GPU Configuration Failed: {e}")

# Run the GPU setup
configure_gpu()

# --------------------------------------------------
# 1. SCENE CLEANUP
# --------------------------------------------------
# Clean the scene (remove Cube, Light, Camera) to avoid clutter
bpy.ops.wm.read_factory_settings(use_empty=True)

input_path = os.path.abspath(INPUT_FILE)
output_path = os.path.abspath(OUTPUT_FILE)

if not os.path.exists(input_path):
    print(f"❌ Error: Could not find {INPUT_FILE}")
    exit()

# --------------------------------------------------
# 2. IMPORT
# --------------------------------------------------
print(f"🔄 Importing {INPUT_FILE}...")

try:
    if INPUT_FILE.lower().endswith(".fbx"):
        bpy.ops.import_scene.fbx(filepath=input_path)
    elif INPUT_FILE.lower().endswith(".obj"):
        bpy.ops.import_scene.obj(filepath=input_path)
    elif INPUT_FILE.lower().endswith(".blend"):
        bpy.ops.wm.open_mainfile(filepath=input_path)
    elif INPUT_FILE.lower().endswith(".glb") or INPUT_FILE.lower().endswith(".gltf"):
        bpy.ops.import_scene.gltf(filepath=input_path)
    else:
        print(f"❌ Format not supported: {INPUT_FILE}")
        exit()
except Exception as e:
    print(f"❌ Import Failed: {e}")
    exit()

print("✅ Import Successful.")

# --------------------------------------------------
# 3. OPTIMIZE & EXPORT
# --------------------------------------------------
print(f"💾 Saving as {OUTPUT_FILE}...")

# Ensure we are in Object Mode
if bpy.context.object and bpy.context.object.mode != 'OBJECT':
    bpy.ops.object.mode_set(mode='OBJECT')

try:
    bpy.ops.export_scene.gltf(
        filepath=output_path,
        export_format='GLB',
        use_selection=False, # Export everything
        export_apply=True,   # Apply modifiers
        export_image_format='AUTO', # Keep textures efficient
        export_draco_mesh_compression_enable=False # Set True if you want smaller files (but slower export)
    )
    print("🚀 Conversion Complete! Ready for App.")
except Exception as e:
    print(f"❌ Export Failed: {e}")