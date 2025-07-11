{
  description = "NVIDIA ACE Pipecat SDK";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs?ref=nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    pyproject-nix = {
      url = "github:pyproject-nix/pyproject.nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    uv2nix = {
      url = "github:pyproject-nix/uv2nix";
      inputs.pyproject-nix.follows = "pyproject-nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    pyproject-build-systems = {
      url = "github:pyproject-nix/build-system-pkgs";
      inputs.pyproject-nix.follows = "pyproject-nix";
      inputs.uv2nix.follows = "uv2nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = {
    self,
    nixpkgs,
    flake-utils,
    uv2nix,
    pyproject-nix,
    pyproject-build-systems,
  }:
    flake-utils.lib.eachSystem ["x86_64-linux"] (
      system: let
        inherit (nixpkgs) lib;
        pkgs = nixpkgs.legacyPackages.${system};
        workspace = uv2nix.lib.workspace.loadWorkspace {workspaceRoot = ./.;};

        overlay = workspace.mkPyprojectOverlay {
          sourcePreference = "wheel";
        };

        # Extend generated overlay with build fixups
        #
        # Uv2nix can only work with what it has, and uv.lock is missing essential metadata to perform some builds.
        # This is an additional overlay implementing build fixups.
        # See:
        # - https://pyproject-nix.github.io/uv2nix/FAQ.html
        pyprojectOverrides = final: prev: {
          numba = prev.numba.overrideAttrs (old: {
            buildInputs = (old.buildInputs or []) ++ [pkgs.tbb_2021_11];
          });
          semantic-version = prev.semantic-version.overrideAttrs (old: {
            nativeBuildInputs =
              old.nativeBuildInputs
              ++ final.resolveBuildSystem {
                setuptools = [];
                wheel = [];
              };
          });

          setuptools-scm = prev.setuptools-scm.overrideAttrs (old: {
            nativeBuildInputs =
              old.nativeBuildInputs
              ++ final.resolveBuildSystem {
                setuptools = [];
                wheel = [];
              };
          });

          setuptools-rust = prev.setuptools-rust.overrideAttrs (old: {
            nativeBuildInputs =
              old.nativeBuildInputs
              ++ final.resolveBuildSystem {
                setuptools = [];
                wheel = [];
              };
          });

          libcst = prev.libcst.overrideAttrs (old: {
            nativeBuildInputs =
              old.nativeBuildInputs
              ++ final.resolveBuildSystem {
                setuptools = [];
                wheel = [];
              };
          });
          nvidia-pipecat = prev.nvidia-pipecat.overrideAttrs (old: {
            nativeBuildInputs =
              old.nativeBuildInputs
              ++ final.resolveBuildSystem {
                hatchling = [];
                editables = [];
              };
          });
        };

        # Use Python 3.12 from nixpkgs
        python = pkgs.python312;

        # Construct package set
        pythonSet =
          # Use base package set from pyproject.nix builders
          (pkgs.callPackage pyproject-nix.build.packages {
            inherit python;
          })
          .overrideScope
          (
            lib.composeManyExtensions [
              pyproject-build-systems.overlays.default
              overlay
              pyprojectOverrides
            ]
          );
        devVenv = pythonSet.mkVirtualEnv "nvidia-pipecat-dev-env" workspace.deps.all;
      in {
        devShells = rec {
          # It is of course perfectly OK to keep using an impure virtualenv workflow and only use uv2nix to build packages.
          # This devShell simply adds Python and undoes the dependency leakage done by Nixpkgs Python infrastructure.
          impure = pkgs.mkShell {
            packages = [
              python
              pkgs.uv
            ];
            shellHook = ''
              unset PYTHONPATH
              export UV_PYTHON_DOWNLOADS=never
            '';
          };

          # This devShell uses uv2nix to construct a virtual environment purely from Nix, using the same dependency specification as the application.
          # The notable difference is that we also apply another overlay here enabling editable mode ( https://setuptools.pypa.io/en/latest/userguide/development_mode.html ).
          #
          # This means that any changes done to your local files do not require a rebuild.
          uv2nix = let
            # Create an overlay enabling editable mode for all local dependencies.
            editableOverlay = workspace.mkEditablePyprojectOverlay {
              # Use environment variable
              root = "$REPO_ROOT";
              # Optional: Only enable editable for these packages
              # members = [ "hello-world" ];
            };

            # Override previous set with our overrideable overlay.
            editablePythonSet = pythonSet.overrideScope editableOverlay;

            # Build virtual environment, with local packages being editable.
            #
            # Enable all optional dependencies for development.
            virtualenv = editablePythonSet.mkVirtualEnv "nvidia-pipecat-dev-env" workspace.deps.all;
          in
            pkgs.mkShell {
              packages = [
                virtualenv
                pkgs.uv
              ];
              shellHook = ''
                # Undo dependency propagation by nixpkgs.
                unset PYTHONPATH

                # Don't create venv using uv
                export UV_NO_SYNC=1

                # Prevent uv from downloading managed Python's
                export UV_PYTHON_DOWNLOADS=never

                # Get repository root using git. This is expanded at runtime by the editable `.pth` machinery.
                export REPO_ROOT=$(git rev-parse --show-toplevel)
              '';
            };
          default = uv2nix;
        };

        checks = {
          lint =
            pkgs.runCommand "lint"
            {
              src = ./.;
              buildInputs = [devVenv];
            }
            ''
              mkdir $out/
              cp -r $src/** .
              ruff check --cache-dir=. -e --output-format=gitlab -o $out/report.json .
            '';
          format =
            pkgs.runCommand "format"
            {
              src = ./.;
              buildInputs = [devVenv];
            }
            ''
              mkdir -p $out/
              echo ok > $out/output
              ruff format --cache-dir=. --diff $src/
            '';
        };

        formatter = pkgs.alejandra;
      }
    );
}
