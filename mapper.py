#!/usr/bin/env python3
"""Code Mapper: bidirectional sync between code and JSON, with strict .gitignore support via pathspec."""

import argparse
import json
import os

import pathspec

DEFAULT_OUTPUT = "project_structure.json"
DEFAULT_ROOT = os.path.dirname(os.path.abspath(__file__))


def read_file_content(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def write_file(file_path: str, content: str) -> None:
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"📄 File created: {file_path}")


def get_gitignore_spec(root_dir: str) -> pathspec.PathSpec:
    """Strict .gitignore matching, plus always-ignored defaults."""
    patterns = [
        ".git/",
        "__pycache__/",
        "*.pyc",
        ".venv/",
        ".DS_Store",
        "node_modules/",
        "*.svg",  # promote .puml, .tikz or mermaid
    ]

    gitignore_path = os.path.join(root_dir, ".gitignore")

    if os.path.exists(gitignore_path):
        with open(gitignore_path, "r", encoding="utf-8") as f:
            patterns.extend(f.readlines())

    return pathspec.PathSpec.from_lines("gitwildmatch", patterns)


def generate_code_from_json(json_path: str) -> None:
    if not os.path.exists(json_path):
        print(f"❌ Error: {json_path} not found.")
        return

    with open(json_path, "r", encoding="utf-8") as f:
        project_data = json.load(f)

    for file_info in project_data["files"]:
        write_file(file_info["path"], file_info["content"])

    print("✅ Code generated successfully!")


def generate_json_from_code(
    root_dir: str,
    output_json_path: str,
    excluded_dirs: list[str] | None = None,
) -> None:
    """Serialize a code directory into JSON, respecting .gitignore and skipping binaries."""
    files = []
    spec = get_gitignore_spec(root_dir)
    excluded_dirs = set(excluded_dirs or [])
    internal_excludes = {".gitignore", "LICENSE", os.path.basename(output_json_path)}

    print(f"🔍 Scanning: {root_dir}")
    print(f"📁 Target: {output_json_path}")
    if excluded_dirs:
        print(f"🚫 Excluded dirs: {', '.join(sorted(excluded_dirs))}")

    for dirpath, dirnames, filenames in os.walk(root_dir):
        relative_dir = os.path.relpath(dirpath, root_dir)
        dirnames[:] = [
            d
            for d in dirnames
            if os.path.relpath(os.path.join(dirpath, d), root_dir).split(os.sep)[0] not in excluded_dirs
        ]
        if relative_dir != "." and spec.match_file(relative_dir):
            dirnames[:] = []
            continue

        for filename in filenames:
            if filename in internal_excludes:
                continue
            file_path = os.path.join(dirpath, filename)
            relative_path = os.path.relpath(file_path, root_dir)
            if spec.match_file(relative_path):
                continue
            try:
                with open(file_path, "tr", encoding="utf-8") as check_file:
                    check_file.read(1024)  # binary check
                files.append({"path": relative_path, "content": read_file_content(file_path)})
            except (UnicodeDecodeError, PermissionError):
                continue
            except OSError as e:
                print(f"⚠️ Error reading {relative_path}: {e}")

    project_data = {"files": files}

    os.makedirs(os.path.dirname(os.path.abspath(output_json_path)), exist_ok=True)

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(project_data, f, indent=2, ensure_ascii=False)

    print(f"✅ JSON generated successfully with {len(files)} files.")


def main():
    parser = argparse.ArgumentParser(description="Code Mapper: Sync code and JSON.")
    parser.add_argument("--from-json", nargs="?", const=DEFAULT_OUTPUT, help="JSON to Code.")
    parser.add_argument("--to-json", nargs="*", help="Code to JSON [ROOT] [OUTPUT].")
    parser.add_argument("--exclude", nargs="*", default=[], help="Directories to exclude (e.g. scripts tests docs).")
    args = parser.parse_args()

    if args.from_json:
        generate_code_from_json(args.from_json)
    elif args.to_json is not None:
        root = args.to_json[0] if len(args.to_json) > 0 else DEFAULT_ROOT
        output = args.to_json[1] if len(args.to_json) > 1 else DEFAULT_OUTPUT
        generate_json_from_code(root, output, excluded_dirs=args.exclude)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
