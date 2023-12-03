import launch

# TODO: add pip dependency if need extra module only on extension

# if not launch.is_installed("aitextgen"):
#     launch.run_pip("install aitextgen==0.6.0", "requirements for MagicPrompt")
if not launch.is_installed("transformers==4.35.2"):
    launch.run_pip("install accelerate==0.20.3", "requirements for Image Caption")
    launch.run_pip("install transformers==4.35.2", "requirements for Image Caption")

if not launch.is_installed("open_clip_torch"):
    launch.run_pip("install open_clip_torch", "requirements for Image Caption")