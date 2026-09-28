"""命令行入口：tuxun add / index / search / serve。"""

import argparse
import logging
import sys

from .config import Config
from .library import Library


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tuxun", description="图寻：用文字搜索本地图片")
    sub = parser.add_subparsers(dest="command", required=True)

    p_add = sub.add_parser("add", help="登记要管理的图片文件夹")
    p_add.add_argument("folders", nargs="+")

    p_rm = sub.add_parser("remove", help="移除已登记的文件夹及其索引")
    p_rm.add_argument("folder")

    sub.add_parser("folders", help="列出已登记的文件夹")
    sub.add_parser("index", help="为已登记的文件夹建立或更新索引")

    p_search = sub.add_parser("search", help="用文字搜索图片")
    p_search.add_argument("text")
    p_search.add_argument("-n", "--limit", type=int, default=20)

    p_serve = sub.add_parser("serve", help="启动本地网页界面")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8765)

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    library = Library(Config.load())

    if args.command == "add":
        for folder in args.folders:
            try:
                print(f"已登记：{library.add_folder(folder)}")
            except ValueError as e:
                print(f"跳过 {folder}：{e}", file=sys.stderr)
        print("运行 `tuxun index` 建立索引")
    elif args.command == "remove":
        library.remove_folder(args.folder)
    elif args.command == "folders":
        for folder in library.folders():
            print(folder)
    elif args.command == "index":
        if not library.folders():
            print("还没有登记文件夹，先运行 `tuxun add <文件夹>`", file=sys.stderr)
            return 1

        def show(p):
            print(f"\r索引中 {p.done}/{p.total}", end="", flush=True)

        result = library.run_index(show)
        print(f"\n完成：新增或更新 {result.added} 张，移除 {result.removed} 张，失败 {len(result.failed)} 张")
    elif args.command == "search":
        for item in library.search_text(args.text, args.limit):
            print(f"{item['score']:.3f}  {item['path']}")
    elif args.command == "serve":
        import uvicorn

        from .server import create_app

        print(f"打开 http://{args.host}:{args.port}")
        uvicorn.run(create_app(library), host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    sys.exit(main())
