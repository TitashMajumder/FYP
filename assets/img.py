import uuid
import argparse
from pathlib import Path

IMG_EXT = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}

def main():
    p = argparse.ArgumentParser(description="Batch rename images to IMG_###.<ext>")
    p.add_argument("folder", type=Path, help="Folder containing images")
    p.add_argument("--start", type=int, default=1, help="Start index (default 1)")
    p.add_argument("--pad", type=int, default=3, help="Number of digits for padding (default 3 => 001)")
    p.add_argument("--prefix", default="IMG_", help="Prefix for new names (default 'IMG_')")
    p.add_argument("--sort", choices=("name","mtime"), default="name", help="Sort by 'name' or 'mtime'")
    p.add_argument("--dryrun", action="store_true", help="Show actions without renaming")
    p.add_argument("--limit", type=int, default=0, help="Optional: stop after this many files (0 => no limit)")
    args = p.parse_args()

    folder = args.folder
    if not folder.is_dir():
        raise SystemExit(f"Error: {folder} is not a directory")

    files = [f for f in folder.iterdir() if f.is_file() and f.suffix.lower() in IMG_EXT]
    if not files:
        print("No image files found.")
        return

    if args.sort == "mtime":
        files.sort(key=lambda x: x.stat().st_mtime)
    else:
        files.sort(key=lambda x: x.name.lower())

    if args.limit and args.limit < len(files):
        files = files[:args.limit]

    mappings = []
    idx = args.start
    for f in files:
        new_name = f"{args.prefix}{str(idx).zfill(args.pad)}{f.suffix.lower()}"
        new_path = folder / new_name
        mappings.append((f, new_path))
        idx += 1

    # collision check
    new_names = [n.name for (_, n) in mappings]
    if len(set(new_names)) != len(new_names):
        print("Name collision detected. Aborting. Try changing prefix/pad/start.")
        return

    # show dry run
    if args.dryrun:
        print("DRY RUN - the following renames would be performed:")
        for old, new in mappings:
            print(f"{old.name} -> {new.name}")
        print(f"\nTotal files: {len(mappings)}")
        return

    # perform rename: use a two-step safe approach for overlapping names
    # 1) rename to temporary unique names
    tmp_map = []
    for old, new in mappings:
        tmp = old.with_name(old.name + f".{uuid.uuid4().hex[:8]}.renametmp")
        old.rename(tmp)
        tmp_map.append((tmp, new))

    # 2) rename temporary -> final
    for tmp, final in tmp_map:
        tmp.rename(final)
        print(f"{tmp.name} -> {final.name}")

    print("Done. Total renamed:", len(tmp_map))

if __name__ == "__main__":
    main()