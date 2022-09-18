import modal

stub = modal.Stub()
stub.image = (
    modal.Image.debian_slim()
    # Install Julia 1.7
    .apt_install(["wget", "ca-certificates"])
    .run_commands(
        [
            "wget -nv https://julialang-s3.julialang.org/bin/linux/x64/1.7/julia-1.7.2-linux-x86_64.tar.gz",
            "tar -xf julia-1.7.2-linux-x86_64.tar.gz",
            "cp -r julia-1.7.2 /opt/",
            "ln -s /opt/julia-1.7.2/bin/julia /usr/local/bin/julia",
        ]
    )
    # Install PyJulia bindings
    .pip_install(["julia"])
    .run_commands(['python -c "import julia; julia.install()"'])
)


@stub.function
def julia_subprocess():
    """Run the Julia interpreter as a subprocess."""
    import subprocess

    print("-> Calling Julia as a subprocess...")
    subprocess.run('julia -e "println(2 + 3)"', shell=True)


@stub.function
def julia_matrix_determinant():
    """Compute the determinant of a random matrix with PyJulia."""
    from julia.Base import rand
    from julia.LinearAlgebra import det

    print("-> Calling Julia using PyJulia...")
    print(det(rand(5, 5)))
    print(det(rand(10, 10)))


if __name__ == "__main__":
    with stub.run():
        julia_subprocess()
        julia_matrix_determinant()
