{ pkgs }: {
  deps = [
    pkgs.ffmpeg-full
    pkgs.python39
    pkgs.python39Packages.pip
    pkgs.nodejs
    pkgs.yarn
  ];

  shellHook = ''
    pip install transformers datasets gradio
  '';
}