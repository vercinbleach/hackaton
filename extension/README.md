# Cala FastPath extension

The extension adds a `FastPath` mode to Cala Knowledge Query. The current planner is a mock
served from localhost. Cala still executes the compiled Cala QL with the signed-in browser
session and creates the normal result URL and Recent entry.

## Run the mock

```powershell
uv run python -m cala_fastpath_training.demo_server
```

Open `http://127.0.0.1:8765/demo/` to test the same content script without installing the
extension.

## Load in Chrome

1. Open `chrome://extensions`.
2. Enable Developer mode.
3. Select Load unpacked.
4. Choose this `extension` directory.
5. Open `https://console.cala.ai/playground/knowledge-query`.

The content script can load across the Cala console so it survives SPA navigation, but it
activates only on Knowledge Query. The extension also accesses the localhost planner and
session storage. It does not read Cala credentials or call the Cala API directly.
