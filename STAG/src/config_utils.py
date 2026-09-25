"""
配置读取的统一入口。

设计约束（都是踩过的坑）：

1. **代码里必须保留同值默认。** 配置缺项不能崩——否则任何一份旧 yaml、
   任何一个漏填的字段都会让整批实验在第一个 episode 挂掉，而 traceback
   还常常被上层的 except 吞掉。

2. **环境变量优先级最高。** 消融要能不改文件就切换，否则并行跑两个配置
   会互相覆盖。优先级：环境变量 > 配置文件 > 代码默认。

3. **取值要留痕。** `describe()` 把实际生效的值打成一行，跟着日志走。
   环境变量跑完就没了，事后对不上是哪一版——这是之前真实吃过的亏。
"""

import os

ENV_PREFIX = "STZS_"      # 历史前缀，与既有运行脚本和日志保持兼容


# ----------------------------------------------------------------------
def _cast_like(raw, default):
    """按默认值的类型解释环境变量字符串。"""
    if default is None:
        return raw
    if isinstance(default, bool):
        return str(raw).strip().lower() not in ("0", "", "false", "no")
    try:
        if isinstance(default, int):
            return int(float(raw))
        if isinstance(default, float):
            return float(raw)
    except (TypeError, ValueError):
        return default
    return raw


class Section:
    """
    配置里的一组同类参数。

        h = Section(config.get('hyper', {}).get('room', {}), 'ROOM')
        self.ROOM_DOOR_CUT = h('door_cut', 0.55)

    取值顺序：环境变量 STZS_ROOM_DOOR_CUT > yaml 的 door_cut > 代码默认 0.55。
    """

    def __init__(self, data, env_group=None):
        self._data = data if isinstance(data, dict) else {}
        self._env_group = (env_group or "").upper()
        self._used = {}       # 记录实际生效的值与来源，供 describe() 打印

    def __call__(self, key, default=None):
        src = "default"
        val = default

        if key in self._data and self._data[key] is not None:
            val, src = self._data[key], "yaml"
            if isinstance(default, (int, float)) and not isinstance(default, bool):
                try:
                    val = type(default)(val)
                except (TypeError, ValueError):
                    val, src = default, "default(bad yaml)"

        env_key = f"{ENV_PREFIX}{self._env_group}_{key}".upper() if self._env_group \
            else f"{ENV_PREFIX}{key}".upper()
        if env_key in os.environ:
            val, src = _cast_like(os.environ[env_key], default), "env"

        self._used[key] = (val, src)
        return val

    def describe(self, only_overridden=True):
        """生效值的一行摘要。默认只打印非代码默认的项，避免刷屏。"""
        items = [(k, v) for k, (v, s) in self._used.items()
                 if not only_overridden or s != "default"]
        if not items:
            return ""
        body = " ".join(f"{k}={v}" for k, v in sorted(items))
        return f"[{self._env_group.lower() or 'cfg'}] {body}"


# ----------------------------------------------------------------------
def hyper(config, group):
    """取 config['hyper'][group] 这一组。config 可以是整份 yaml，也可以是 sim_cfg。"""
    root = config.get("hyper", {}) if isinstance(config, dict) else {}
    return Section((root or {}).get(group, {}) or {}, group)


def paths(config):
    """
    取 paths 段。每一项都是完整绝对路径，代码里不做拼接——
    这样数据集散在不同盘上也能配，代价是换机器要多改几行。
    """
    return Section((config or {}).get("paths", {}) or {}, "PATH")


def resolve_path(cfg_paths, key, default=None, must_exist=False):
    """
    读一条路径。`must_exist=True` 时找不到就抛异常并说清是哪一项——
    比让 habitat 在几十行之后报一个 'scene not found' 好定位得多。
    """
    val = cfg_paths(key, default)
    if val and val.startswith("~"):
        val = os.path.expanduser(val)
    if must_exist and (not val or not os.path.exists(val)):
        raise FileNotFoundError(
            f"配置项 paths.{key} 指向的路径不存在：{val!r}\n"
            f"  请在 config/*.yaml 的 paths: 段填写正确的绝对路径，"
            f"或设置环境变量 {ENV_PREFIX}PATH_{key.upper()}"
        )
    return val
