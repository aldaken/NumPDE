-- NPDE project-local config (loaded via exrc)

local root = vim.fn.fnamemodify(debug.getinfo(1, "S").source:sub(2), ":p:h")
local debug_dir = root .. "/build-debug"
local release_dir = root .. "/build"
local nproc = "-j" .. tostring(vim.uv.available_parallelism())

-- Detect dir/problem/solution from current buffer path
local function get_info()
  local path = vim.fn.expand("%:p")
  for _, d in ipairs({ "developers", "homeworks", "lecturecodes" }) do
    local dir, prob, sol = path:match("/(" .. d .. ")/([^/]+)/([^/]+)/")
    if dir then return { dir = dir, problem = prob, solution = sol } end
  end
end

local function target(info)
  local sfx = info.dir == "developers" and "_dev" or ""
  return info.problem .. "_" .. info.solution .. sfx
end

local function binary_path(bdir, info)
  return ("%s/%s/%s/%s_%s"):format(bdir, info.dir, info.problem, info.problem, info.solution)
end

-- Run a command in a small terminal split
local function term_run(cmd)
  vim.cmd("botright 15split | terminal " .. cmd)
  vim.cmd("startinsert")
end

-- Keymaps
vim.keymap.set("n", "<leader>mb", function()
  local info = get_info()
  if not info then return end
  term_run(("cmake --build %s --target %s %s"):format(debug_dir, target(info), nproc))
end, { desc = "Make build (debug)" })

vim.keymap.set("n", "<leader>mB", function()
  local info = get_info()
  if not info then return end
  term_run(("cmake --build %s --target %s %s"):format(release_dir, target(info), nproc))
end, { desc = "Make release" })

vim.keymap.set("n", "<leader>mc", function()
  term_run(("cmake -B %s -DCMAKE_BUILD_TYPE=Debug %s"):format(debug_dir, root))
end, { desc = "Make configure (debug)" })

vim.keymap.set("n", "<leader>mC", function()
  term_run(("cmake -B %s %s"):format(release_dir, root))
end, { desc = "Make configure (release)" })

-- DAP config: inject when a C++ file is first opened
local dap_ready = false
vim.api.nvim_create_autocmd("FileType", {
  pattern = { "c", "cpp" },
  callback = function()
    if dap_ready then return end
    vim.defer_fn(function()
      local ok, dap = pcall(require, "dap")
      if not ok then return end
      dap_ready = true

      local existing = dap.configurations.cpp or {}
      dap.configurations.cpp = vim.list_extend({
        {
          name = "NPDE: Debug current problem",
          type = "codelldb", request = "launch",
          initCommands = { "command script import " .. root .. "/scripts/lldb_eigen_printer.py" },
          program = function()
            local info = get_info()
            if not info then return dap.ABORT end
            return binary_path(debug_dir, info)
          end,
          cwd = function()
            local info = get_info()
            if not info then return debug_dir end
            return ("%s/%s/%s"):format(debug_dir, info.dir, info.problem)
          end,
        },
      }, existing)
    end, 100)
  end,
})
