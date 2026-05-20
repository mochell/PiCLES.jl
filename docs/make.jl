using Documenter
using PiCLES

DocMeta.setdocmeta!(PiCLES, :DocTestSetup, :(using PiCLES); recursive=true)

makedocs(;
    modules = [PiCLES],
    authors = "Momme C. Hell and contributors",
    sitename = "PiCLES.jl",
    format = Documenter.HTML(;
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical = "https://mochell.github.io/PiCLES.jl",
        edit_link = "main",
        assets = String[],
    ),
    pages = [
        "Home" => "index.md",
        "Model" => "model.md",
        "Quick start" => "quickstart.md",
        "API reference" => "api.md",
    ],
    warnonly = true,
)

deploydocs(;
    repo = "github.com/mochell/PiCLES.jl",
    devbranch = "main",
    push_preview = true,
)
