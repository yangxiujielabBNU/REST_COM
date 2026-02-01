"""
变量血缘分析工具 - 分析 Jupyter Notebook 中变量的依赖关系
用法: python variable_lineage.py <notebook.ipynb> [目标变量]
"""

import ast
import json
import sys
from collections import defaultdict
from pathlib import Path


class VariableLineageAnalyzer(ast.NodeVisitor):
    """AST 访问器，提取变量赋值和依赖关系"""

    def __init__(self):
        # 变量依赖关系: {变量名: {依赖的变量集合}}
        self.dependencies = defaultdict(set)
        # 变量定义位置: {变量名: cell编号}
        self.defined_in_cell = {}
        # 当前分析的 cell 编号
        self.current_cell = 0
        # 函数定义: {函数名: 参数列表}
        self.functions = {}
        # 当前正在收集的依赖变量
        self._current_deps = set()

    def set_cell(self, cell_num):
        """设置当前 cell 编号"""
        self.current_cell = cell_num

    def visit_Import(self, node):
        """处理 import 语句: import x, import x as y"""
        for alias in node.names:
            name = alias.asname if alias.asname else alias.name.split('.')[0]
            self.defined_in_cell[name] = f"import@{self.current_cell}"

    def visit_ImportFrom(self, node):
        """处理 from x import y 语句"""
        for alias in node.names:
            name = alias.asname if alias.asname else alias.name
            self.defined_in_cell[name] = f"import@{self.current_cell}"

    def visit_FunctionDef(self, node):
        """记录函数定义"""
        self.functions[node.name] = [arg.arg for arg in node.args.args]
        self.defined_in_cell[node.name] = self.current_cell
        # 继续访问函数体
        self.generic_visit(node)

    def visit_Assign(self, node):
        """处理赋值语句: x = expr"""
        # 收集右侧表达式中的变量依赖
        deps = self._extract_dependencies(node.value)

        # 处理所有赋值目标
        for target in node.targets:
            self._process_target(target, deps)

        self.generic_visit(node)

    def visit_AugAssign(self, node):
        """处理增量赋值: x += expr"""
        deps = self._extract_dependencies(node.value)

        if isinstance(node.target, ast.Name):
            var_name = node.target.id
            # 增量赋值依赖自身和右侧
            deps.add(var_name)
            self.dependencies[var_name].update(deps)
            self.defined_in_cell[var_name] = self.current_cell

        self.generic_visit(node)

    def visit_For(self, node):
        """处理 for 循环"""
        # 循环变量依赖于迭代对象
        iter_deps = self._extract_dependencies(node.iter)

        if isinstance(node.target, ast.Name):
            var_name = node.target.id
            self.dependencies[var_name].update(iter_deps)
            self.defined_in_cell[var_name] = self.current_cell
        elif isinstance(node.target, ast.Tuple):
            for elt in node.target.elts:
                if isinstance(elt, ast.Name):
                    self.dependencies[elt.id].update(iter_deps)
                    self.defined_in_cell[elt.id] = self.current_cell

        self.generic_visit(node)

    def visit_With(self, node):
        """处理 with 语句: with open(path) as f"""
        for item in node.items:
            # item.context_expr 是 with 后面的表达式 (如 open(pickle_path, 'rb'))
            # item.optional_vars 是 as 后面的变量 (如 f)
            if item.optional_vars:
                deps = self._extract_dependencies(item.context_expr)
                self._process_target(item.optional_vars, deps)

        # 继续访问 with 块内的代码
        self.generic_visit(node)

    def _process_target(self, target, deps):
        """处理赋值目标"""
        if isinstance(target, ast.Name):
            # 简单赋值: x = ...
            var_name = target.id
            self.dependencies[var_name].update(deps)
            self.defined_in_cell[var_name] = self.current_cell

        elif isinstance(target, ast.Tuple) or isinstance(target, ast.List):
            # 解包赋值: x, y = ...
            for elt in target.elts:
                if isinstance(elt, ast.Name):
                    self.dependencies[elt.id].update(deps)
                    self.defined_in_cell[elt.id] = self.current_cell

        elif isinstance(target, ast.Subscript):
            # 索引赋值: x[i] = ...
            if isinstance(target.value, ast.Name):
                var_name = target.value.id
                # 索引赋值时，变量依赖自身和右侧
                self.dependencies[var_name].add(var_name)
                self.dependencies[var_name].update(deps)

    def _extract_dependencies(self, node):
        """从表达式中提取变量依赖"""
        deps = set()

        if isinstance(node, ast.Name):
            deps.add(node.id)

        elif isinstance(node, ast.Call):
            # 函数调用: func(args)
            # 添加函数名作为依赖（如果是变量）
            if isinstance(node.func, ast.Name):
                deps.add(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                # 方法调用: obj.method()
                deps.update(self._extract_dependencies(node.func.value))

            # 添加参数依赖
            for arg in node.args:
                deps.update(self._extract_dependencies(arg))
            for kw in node.keywords:
                deps.update(self._extract_dependencies(kw.value))

        elif isinstance(node, ast.BinOp):
            # 二元运算: a + b
            deps.update(self._extract_dependencies(node.left))
            deps.update(self._extract_dependencies(node.right))

        elif isinstance(node, ast.UnaryOp):
            # 一元运算: -x
            deps.update(self._extract_dependencies(node.operand))

        elif isinstance(node, ast.Compare):
            # 比较运算: a < b
            deps.update(self._extract_dependencies(node.left))
            for comp in node.comparators:
                deps.update(self._extract_dependencies(comp))

        elif isinstance(node, ast.Subscript):
            # 索引访问: x[i]
            deps.update(self._extract_dependencies(node.value))
            deps.update(self._extract_dependencies(node.slice))

        elif isinstance(node, ast.Attribute):
            # 属性访问: x.attr
            deps.update(self._extract_dependencies(node.value))

        elif isinstance(node, ast.List) or isinstance(node, ast.Tuple):
            # 列表/元组: [a, b, c]
            for elt in node.elts:
                deps.update(self._extract_dependencies(elt))

        elif isinstance(node, ast.Dict):
            # 字典: {k: v}
            for key in node.keys:
                if key:
                    deps.update(self._extract_dependencies(key))
            for val in node.values:
                deps.update(self._extract_dependencies(val))

        elif isinstance(node, ast.ListComp) or isinstance(node, ast.DictComp):
            # 列表/字典推导式
            deps.update(self._extract_comprehension_deps(node))

        elif isinstance(node, ast.IfExp):
            # 三元表达式: a if cond else b
            deps.update(self._extract_dependencies(node.test))
            deps.update(self._extract_dependencies(node.body))
            deps.update(self._extract_dependencies(node.orelse))

        elif isinstance(node, ast.Index):
            # Python 3.8 以下的索引
            deps.update(self._extract_dependencies(node.value))

        elif isinstance(node, ast.Slice):
            # 切片: x[a:b:c]
            if node.lower:
                deps.update(self._extract_dependencies(node.lower))
            if node.upper:
                deps.update(self._extract_dependencies(node.upper))
            if node.step:
                deps.update(self._extract_dependencies(node.step))

        return deps

    def _extract_comprehension_deps(self, node):
        """提取推导式的依赖"""
        deps = set()

        if isinstance(node, ast.ListComp):
            deps.update(self._extract_dependencies(node.elt))
        elif isinstance(node, ast.DictComp):
            deps.update(self._extract_dependencies(node.key))
            deps.update(self._extract_dependencies(node.value))

        for generator in node.generators:
            deps.update(self._extract_dependencies(generator.iter))
            for if_clause in generator.ifs:
                deps.update(self._extract_dependencies(if_clause))

        return deps


def parse_notebook(notebook_path):
    """解析 Jupyter Notebook，提取代码单元"""
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    cells = []
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            # 跳过 magic 命令和空单元
            lines = [line for line in source.split('\n')
                     if not line.strip().startswith('%')
                     and not line.strip().startswith('!')]
            code = '\n'.join(lines)
            if code.strip():
                cells.append((i, code))

    return cells


def analyze_notebook(notebook_path):
    """分析 notebook 中的变量血缘关系"""
    cells = parse_notebook(notebook_path)
    analyzer = VariableLineageAnalyzer()

    for cell_num, code in cells:
        analyzer.set_cell(cell_num)
        try:
            tree = ast.parse(code)
            analyzer.visit(tree)
        except SyntaxError as e:
            print(f"Cell {cell_num} 语法错误: {e}")
            continue

    return analyzer


def filter_internal_variables(dependencies):
    """过滤掉内置变量和模块"""
    builtins = {
        'print', 'len', 'range', 'enumerate', 'zip', 'map', 'filter',
        'int', 'float', 'str', 'list', 'dict', 'set', 'tuple', 'bool',
        'True', 'False', 'None', 'open', 'sum', 'min', 'max', 'abs',
        'sorted', 'reversed', 'any', 'all', 'isinstance', 'type',
        'np', 'pd', 'plt', 'mne', 'os', 'op', 'pickle', 'tqdm',
        'matplotlib', 'numpy', 'pandas', 'scipy',
    }

    filtered = {}
    for var, deps in dependencies.items():
        if var not in builtins and not var.startswith('_'):
            filtered_deps = {d for d in deps if d not in builtins and not d.startswith('_')}
            if filtered_deps or var in dependencies:
                filtered[var] = filtered_deps

    return filtered


def trace_lineage(dependencies, target_var):
    """追溯目标变量的完整血缘链"""
    lineage = set()
    to_visit = [target_var]
    visited = set()

    while to_visit:
        var = to_visit.pop()
        if var in visited:
            continue
        visited.add(var)

        if var in dependencies:
            for dep in dependencies[var]:
                lineage.add((dep, var))
                to_visit.append(dep)

    return lineage


def generate_mermaid(dependencies, target_var=None, title="变量血缘图"):
    """生成 Mermaid 格式的流程图"""
    if target_var:
        edges = trace_lineage(dependencies, target_var)
    else:
        edges = set()
        for var, deps in dependencies.items():
            for dep in deps:
                edges.add((dep, var))

    if not edges:
        return f"```mermaid\ngraph LR\n    empty[无依赖关系]\n```"

    mermaid = f"```mermaid\ngraph LR\n"
    mermaid += f"    subgraph {title}\n"

    for src, dst in sorted(edges):
        # 转义特殊字符
        src_safe = src.replace('[', '_').replace(']', '_')
        dst_safe = dst.replace('[', '_').replace(']', '_')
        mermaid += f"    {src_safe}[{src}] --> {dst_safe}[{dst}]\n"

    mermaid += "    end\n```"
    return mermaid


def generate_graphviz(dependencies, target_var=None, title="Variable Lineage"):
    """生成 Graphviz DOT 格式"""
    if target_var:
        edges = trace_lineage(dependencies, target_var)
    else:
        edges = set()
        for var, deps in dependencies.items():
            for dep in deps:
                edges.add((dep, var))

    dot = f'digraph "{title}" {{\n'
    dot += '    rankdir=LR;\n'
    dot += '    node [shape=box, style=rounded];\n'

    # 如果有目标变量，高亮显示
    if target_var:
        dot += f'    "{target_var}" [style="rounded,filled", fillcolor=lightblue];\n'

    for src, dst in sorted(edges):
        dot += f'    "{src}" -> "{dst}";\n'

    dot += '}\n'
    return dot


def generate_html_report(dependencies, defined_in_cell, target_var=None):
    """生成 HTML 可视化报告"""

    # 获取所有边（完整图）
    all_edges = set()
    for var, deps in dependencies.items():
        for dep in deps:
            all_edges.add((dep, var))

    # 如果有目标变量，获取相关血缘链
    if target_var:
        related_edges = trace_lineage(dependencies, target_var)
        title = f"变量 '{target_var}' 的血缘关系"
        # 收集相关节点
        related_nodes = set()
        for src, dst in related_edges:
            related_nodes.add(src)
            related_nodes.add(dst)
    else:
        related_edges = all_edges
        related_nodes = None
        title = "完整变量血缘图"

    # 收集所有节点
    all_nodes = set()
    for src, dst in all_edges:
        all_nodes.add(src)
        all_nodes.add(dst)

    # 生成节点的 JavaScript 数据
    nodes_js = []
    for i, node in enumerate(sorted(all_nodes)):
        cell = defined_in_cell.get(node, "?")  # 未知的显示 ?
        # 处理 import@N 格式
        if isinstance(cell, str) and cell.startswith("import@"):
            cell_num = int(cell.split("@")[1])
            cell_label = f"import@{cell_num}"
        elif isinstance(cell, int):
            cell_num = cell
            cell_label = str(cell)
        else:
            cell_num = 999
            cell_label = "?"

        if target_var:
            if node == target_var:
                # 目标变量：红色，大号
                color = '{"background": "#FF4444", "border": "#CC0000"}'
                font = '{"size": 14, "color": "#FFFFFF", "bold": true}'
                opacity = 1.0
            elif related_nodes and node in related_nodes:
                # 相关变量：蓝色，正常
                color = '{"background": "#4A90D9", "border": "#2E6DB4"}'
                font = '{"size": 12, "color": "#FFFFFF"}'
                opacity = 1.0
            else:
                # 不相关变量：浅灰色，半透明
                color = '{"background": "#E0E0E0", "border": "#CCCCCC"}'
                font = '{"size": 10, "color": "#999999"}'
                opacity = 0.3
        else:
            # 无目标变量时，所有节点正常显示
            color = '{"background": "#4A90D9", "border": "#2E6DB4"}'
            font = '{"size": 12, "color": "#FFFFFF"}'
            opacity = 1.0

        # 添加 level 属性，按 Cell 编号排列（早出现的在上面）
        nodes_js.append(
            f'{{id: {i}, label: "{node}\\n(Cell {cell_label})", '
            f'level: {cell_num}, '
            f'color: {color}, font: {font}, opacity: {opacity}}}'
        )

    node_id_map = {node: i for i, node in enumerate(sorted(all_nodes))}

    # 生成边的 JavaScript 数据
    edges_js = []
    for src, dst in all_edges:
        if target_var:
            if (src, dst) in related_edges:
                # 相关边：蓝色实线
                edge_color = '{"color": "#4A90D9", "opacity": 1.0}'
                width = 2
            else:
                # 不相关边：浅灰色虚线
                edge_color = '{"color": "#DDDDDD", "opacity": 0.3}'
                width = 1
        else:
            edge_color = '{"color": "#4A90D9", "opacity": 1.0}'
            width = 2

        edges_js.append(
            f'{{from: {node_id_map[src]}, to: {node_id_map[dst]}, '
            f'arrows: "to", color: {edge_color}, width: {width}}}'
        )

    # 生成依赖关系的 JSON（用于前端动态计算血缘链）
    deps_json = json.dumps({var: list(deps) for var, deps in dependencies.items()})
    name_to_id_json = json.dumps(node_id_map)
    id_to_name_json = json.dumps({v: k for k, v in node_id_map.items()})

    html = f'''<!DOCTYPE html>
<html>
<head>
    <title>{title}</title>
    <script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
    <style>
        body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 0; padding: 20px; background: #f8f9fa; }}
        h1 {{ color: #333; margin-bottom: 10px; }}
        .container {{ max-width: 1400px; margin: 0 auto; }}
        .toolbar {{ display: flex; gap: 15px; margin-bottom: 15px; align-items: center; flex-wrap: wrap; }}
        .search-box {{ display: flex; gap: 8px; align-items: center; }}
        .search-box input {{ padding: 8px 12px; border: 1px solid #ddd; border-radius: 6px; font-size: 14px; width: 200px; }}
        .search-box input:focus {{ outline: none; border-color: #4A90D9; box-shadow: 0 0 0 2px rgba(74,144,217,0.2); }}
        .search-box button {{ padding: 8px 16px; background: #4A90D9; color: white; border: none; border-radius: 6px; cursor: pointer; font-size: 14px; }}
        .search-box button:hover {{ background: #3a7bc8; }}
        .search-box button.reset {{ background: #6c757d; }}
        .search-box button.reset:hover {{ background: #5a6268; }}
        .search-results {{ font-size: 13px; color: #666; padding: 5px 10px; background: #e9ecef; border-radius: 4px; display: none; }}
        #network {{ width: 100%; height: 700px; border: 1px solid #ddd; border-radius: 8px; background: #fff; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }}
        .info {{ margin-top: 15px; padding: 15px; background: #fff; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }}
        .legend {{ display: flex; gap: 20px; margin-top: 10px; flex-wrap: wrap; }}
        .legend-item {{ display: flex; align-items: center; gap: 8px; }}
        .legend-color {{ width: 20px; height: 20px; border-radius: 4px; }}
        .target {{ background: #FF4444; }}
        .related {{ background: #4A90D9; }}
        .unrelated {{ background: #E0E0E0; opacity: 0.5; }}
        .found {{ background: #28a745; }}
        .stats {{ display: flex; gap: 30px; margin-bottom: 10px; }}
        .stat {{ padding: 8px 15px; background: #e9ecef; border-radius: 4px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>{title}</h1>
        <div class="toolbar">
            <div class="search-box">
                <input type="text" id="searchInput" placeholder="搜索变量名..." onkeypress="if(event.key==='Enter') searchNode()">
                <button onclick="searchNode()">搜索</button>
                <button class="reset" onclick="resetSearch()">重置</button>
            </div>
            <div class="search-results" id="searchResults"></div>
        </div>
        <div id="network"></div>
        <div class="info">
            <div class="stats">
                <div class="stat"><strong>总节点数:</strong> {len(all_nodes)}</div>
                <div class="stat"><strong>相关节点:</strong> <span id="relatedCount">{len(related_nodes) if related_nodes else len(all_nodes)}</span></div>
                <div class="stat"><strong>总边数:</strong> {len(all_edges)}</div>
            </div>
            <div class="legend">
                <div class="legend-item"><div class="legend-color target"></div><span>目标变量</span></div>
                <div class="legend-item"><div class="legend-color related"></div><span>相关变量（血缘链）</span></div>
                <div class="legend-item"><div class="legend-color unrelated"></div><span>不相关变量</span></div>
                <div class="legend-item"><div class="legend-color found"></div><span>搜索结果</span></div>
            </div>
            <p style="margin-top: 10px; color: #666;"><strong>提示:</strong> 拖拽节点可调整布局，滚轮缩放，点击节点可查看详情</p>
        </div>
    </div>
    <script>
        var nodes = new vis.DataSet([{', '.join(nodes_js)}]);
        var edges = new vis.DataSet([{', '.join(edges_js)}]);
        var container = document.getElementById('network');
        var data = {{ nodes: nodes, edges: edges }};
        var options = {{
            layout: {{
                improvedLayout: false,
                hierarchical: false
            }},
            physics: {{
                enabled: true,
                solver: 'forceAtlas2Based',
                forceAtlas2Based: {{
                    gravitationalConstant: -30,
                    centralGravity: 0.005,
                    springLength: 150,
                    springConstant: 0.05,
                    damping: 0.9
                }},
                stabilization: {{
                    enabled: true,
                    iterations: 150,
                    updateInterval: 25,
                    fit: true
                }},
                maxVelocity: 50,
                minVelocity: 0.75
            }},
            nodes: {{
                shape: "box",
                borderWidth: 1,
                shadow: false,
                margin: 6,
                font: {{ size: 10 }}
            }},
            edges: {{
                smooth: false,
                shadow: false,
                width: 1,
                arrows: {{ to: {{ scaleFactor: 0.5 }} }}
            }},
            interaction: {{
                hover: true,
                tooltipDelay: 300,
                dragNodes: true,
                dragView: true,
                zoomView: true,
                hideEdgesOnDrag: true,
                hideEdgesOnZoom: true
            }}
        }};
        var network = new vis.Network(container, data, options);

        // 稳定后停止物理模拟，提高交互性能
        network.once('stabilized', function() {{
            network.setOptions({{ physics: {{ enabled: false }} }});
        }});

        // 保存原始节点样式用于重置
        var originalNodes = {{}};
        nodes.forEach(function(node) {{
            originalNodes[node.id] = {{
                color: node.color,
                font: node.font,
                opacity: node.opacity
            }};
        }});

        // 变量依赖关系数据（用于动态计算血缘链）
        var dependencies = {deps_json};
        var nodeNameToId = {name_to_id_json};
        var nodeIdToName = {id_to_name_json};

        // 追溯血缘链（限制层级深度）
        function traceLineage(targetVar, maxDepth) {{
            maxDepth = maxDepth || 2;  // 默认只追溯2级上游
            var lineage = new Set();
            var toVisit = [{{varName: targetVar, depth: 0}}];
            var visited = new Set();

            while (toVisit.length > 0) {{
                var current = toVisit.pop();
                var varName = current.varName;
                var depth = current.depth;

                if (visited.has(varName)) continue;
                visited.add(varName);

                if (depth < maxDepth && dependencies[varName]) {{
                    dependencies[varName].forEach(function(dep) {{
                        lineage.add(dep + '->' + varName);
                        toVisit.push({{varName: dep, depth: depth + 1}});
                    }});
                }}
            }}
            return lineage;
        }}

        // 高亮目标变量及其上游来源（只显示来源，不显示去向）
        function highlightLineage(targetVar) {{
            var lineage = traceLineage(targetVar);
            var relatedNodes = new Set();

            // 只收集上游来源节点（不包括目标变量本身，它单独处理）
            lineage.forEach(function(edge) {{
                var parts = edge.split('->');
                relatedNodes.add(parts[0]);  // 只添加来源节点
            }});

            // 更新所有节点样式
            nodes.forEach(function(node) {{
                var nodeName = nodeIdToName[node.id];
                if (nodeName === targetVar) {{
                    // 目标变量：红色
                    nodes.update({{
                        id: node.id,
                        color: {{"background": "#FF4444", "border": "#CC0000"}},
                        font: {{"size": 14, "color": "#FFFFFF", "bold": true}},
                        opacity: 1.0
                    }});
                }} else if (relatedNodes.has(nodeName)) {{
                    // 上游来源变量：蓝色
                    nodes.update({{
                        id: node.id,
                        color: {{"background": "#4A90D9", "border": "#2E6DB4"}},
                        font: {{"size": 12, "color": "#FFFFFF"}},
                        opacity: 1.0
                    }});
                }} else {{
                    // 不相关变量：浅灰色半透明
                    nodes.update({{
                        id: node.id,
                        color: {{"background": "#E0E0E0", "border": "#CCCCCC"}},
                        font: {{"size": 10, "color": "#999999"}},
                        opacity: 0.3
                    }});
                }}
            }});

            // 更新边的样式 - 只高亮上游来源的边
            edges.forEach(function(edge) {{
                var fromName = nodeIdToName[edge.from];
                var toName = nodeIdToName[edge.to];
                var edgeKey = fromName + '->' + toName;

                if (lineage.has(edgeKey)) {{
                    // 上游来源边：蓝色高亮
                    edges.update({{
                        id: edge.id,
                        color: {{"color": "#4A90D9", "opacity": 1.0}},
                        width: 2,
                        hidden: false
                    }});
                }} else {{
                    // 其他边：隐藏
                    edges.update({{
                        id: edge.id,
                        hidden: true
                    }});
                }}
            }});

            // 更新标题
            document.querySelector('h1').textContent = "变量 '" + targetVar + "' 的上游来源";
            document.getElementById('relatedCount').textContent = relatedNodes.size + 1;
        }}

        // 重置为初始状态
        function resetToOriginal() {{
            Object.keys(originalNodes).forEach(function(id) {{
                nodes.update({{
                    id: parseInt(id),
                    color: originalNodes[id].color,
                    font: originalNodes[id].font,
                    opacity: originalNodes[id].opacity
                }});
            }});

            // 重置边（恢复显示）
            edges.forEach(function(edge) {{
                edges.update({{
                    id: edge.id,
                    color: {{"color": "#4A90D9", "opacity": 1.0}},
                    width: 2,
                    hidden: false
                }});
            }});

            document.querySelector('h1').textContent = "{title}";
            document.getElementById('relatedCount').textContent = '{len(related_nodes) if related_nodes else len(all_nodes)}';
            network.unselectAll();
        }}

        // 搜索节点功能 - 只高亮匹配节点，不改变视图
        function searchNode() {{
            var query = document.getElementById('searchInput').value.toLowerCase().trim();
            if (!query) return;

            // 先重置
            resetToOriginal();

            var foundNodes = [];
            var allNodesList = nodes.get();

            // 只高亮匹配的节点
            allNodesList.forEach(function(node) {{
                var label = node.label.split('\\n')[0].toLowerCase();
                if (label.includes(query)) {{
                    foundNodes.push(node);
                    // 高亮找到的节点：绿色
                    nodes.update({{
                        id: node.id,
                        color: {{"background": "#28a745", "border": "#1e7e34"}},
                        font: {{"size": 14, "color": "#FFFFFF", "bold": true}},
                        opacity: 1.0
                    }});
                }}
            }});

            // 显示搜索结果
            var resultsDiv = document.getElementById('searchResults');
            if (foundNodes.length > 0) {{
                resultsDiv.style.display = 'block';
                resultsDiv.innerHTML = '找到 <strong>' + foundNodes.length + '</strong> 个匹配: ' +
                    foundNodes.map(n => n.label.split('\\n')[0]).join(', ');
                // 选中找到的节点（不缩放）
                network.selectNodes(foundNodes.map(n => n.id));
            }} else {{
                resultsDiv.style.display = 'block';
                resultsDiv.innerHTML = '未找到匹配的变量';
            }}
        }}

        // 重置搜索
        function resetSearch() {{
            document.getElementById('searchInput').value = '';
            document.getElementById('searchResults').style.display = 'none';
            resetToOriginal();
        }}

        // 点击节点时高亮其血缘链
        network.on("click", function(params) {{
            if (params.nodes.length > 0) {{
                var nodeId = params.nodes[0];
                var nodeName = nodeIdToName[nodeId];
                highlightLineage(nodeName);
            }}
        }});

        // 双击空白处重置
        network.on("doubleClick", function(params) {{
            if (params.nodes.length === 0) {{
                resetToOriginal();
            }}
        }});

        // 暴露函数到全局
        window.searchNode = searchNode;
        window.resetSearch = resetSearch;
        window.resetToOriginal = resetToOriginal;
    </script>
</body>
</html>'''

    return html


def main():
    if len(sys.argv) < 2:
        print("用法: python variable_lineage.py <notebook.ipynb> [目标变量]")
        print("示例: python variable_lineage.py analysis.ipynb psds_group")
        sys.exit(1)

    notebook_path = sys.argv[1]
    target_var = sys.argv[2] if len(sys.argv) > 2 else None

    print(f"分析文件: {notebook_path}")
    if target_var:
        print(f"目标变量: {target_var}")

    # 分析 notebook
    analyzer = analyze_notebook(notebook_path)

    # 过滤内置变量
    filtered_deps = filter_internal_variables(analyzer.dependencies)

    print(f"\n发现 {len(filtered_deps)} 个变量")

    # 生成 Mermaid 图
    mermaid = generate_mermaid(filtered_deps, target_var)
    print("\n" + "=" * 50)
    print("Mermaid 格式（可粘贴到 Markdown）:")
    print("=" * 50)
    print(mermaid)

    # 生成 HTML 报告
    html = generate_html_report(filtered_deps, analyzer.defined_in_cell, target_var)
    output_path = Path(notebook_path).stem + "_lineage.html"
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"\n✓ HTML 可视化报告已保存: {output_path}")

    # 生成 Graphviz DOT
    dot = generate_graphviz(filtered_deps, target_var)
    dot_path = Path(notebook_path).stem + "_lineage.dot"
    with open(dot_path, 'w', encoding='utf-8') as f:
        f.write(dot)
    print(f"✓ Graphviz DOT 文件已保存: {dot_path}")


if __name__ == '__main__':
    main()
