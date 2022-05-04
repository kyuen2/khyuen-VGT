logo = "
   __           _____          _     
  / /  __ _ ___/__   \\___  ___| |__  
 / /  / _` |_  / / /\\/ _ \\/ __| '_ \\ 
/ /__| (_| |/ / / / |  __/ (__| | | |
\\____/\\__,_/___|\\/   \\___|\\___|_| |_|
"

# LazTech - Project Minimal
# @File: startup.jl
# @Description: Configurate the startup process to automate the loading

# +++++++++ ++++ ++ +++++++++ ++++ ++++++ ++++++++
# Uncomment this to customize your module location
  # LAZMODULEPATH = pwd()
  LAZMODULEPATH = pwd()  			# Default with your current directory
# - - - - - - - - - - - - - - - - - - 


# 0. Check Startup Version
const red = "\e[92m"
if VERSION < v"1.0.0"
	println("\e[91m$logo")
	println("\e[91mAbort: Julia > 1.0.0 version required for the minimal project !")
	println("\e[91mDownload Julia 1.0.3: https://julialang.org/downloads/")
	exit()
end
println(red * logo * "\e[0m")



### 1. Load the startup file in current work directory

push!(LOAD_PATH, LAZMODULEPATH)

println(red * "Loaded LazTech Module : $LAZMODULEPATH")

### 2. Optionall, you can automatically use Laz Modules at startup

# Uncomment these to automatically start Laz Modules at startup
using LazIO
using LazType
using LazCore
using LazPCA

