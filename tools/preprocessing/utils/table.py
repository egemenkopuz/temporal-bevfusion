from typing import List, Optional


def log_table(
    title: str,
    classes: List[str],
    splits: List[str],
    counts,
    ratios,
    levels: Optional[List[str]] = None,
    classname_start_idx: int = 0,
    include_total: bool = False,
    table_header_title_prefix: str = "",
    table_subheader_title_prefix: str = "",
) -> str:
    out = []
    table_subheader_title = create_table_subheader_title(
        table_subheader_title_prefix, splits, levels
    )
    line_length = len(table_subheader_title)
    out.append(create_header_line(line_length, "="))
    out.append(title)
    out.append(create_header_line(line_length, "="))
    if levels is not None:
        out.append(create_table_header_title(table_header_title_prefix, splits, levels))
    out.append(table_subheader_title)
    out.append(create_header_line(line_length, "-"))
    if levels is None:
        out.extend(
            create_class_table(
                classes, splits, counts, ratios, classname_start_idx, include_total, line_length
            )
        )
    else:
        out.extend(
            create_class_level_table(
                classes,
                splits,
                levels,
                counts,
                ratios,
                classname_start_idx,
                include_total,
                line_length,
            )
        )
    return "\n".join(out)


def create_table_header_title(title_prefix: str, splits: List[str], levels: List[str]) -> str:
    title = f"{title_prefix:<20} |"
    split_len = len(splits)
    for level in levels:
        if split_len == 3:
            title += "".join(f" {'':<15} {level:<15} {'':<15}")
        else:
            title += "".join(f" {level:<15} {'':<15}")
        title += "|"
    return title


def create_table_subheader_title(title_prefix: str, splits: List[str], levels: List[str]) -> str:
    title = f"{title_prefix:<20} |"
    split_len = len(splits)
    if levels is None:
        levels = range(1)
    for _ in levels:
        title += "".join([f" {splits[i]:<15}" for i in range(split_len)])
        title += "|"
    return title


def create_header_line(length: int, symbol: str = "-") -> str:
    return symbol * length


def create_class_table(
    classes,
    splits,
    counts,
    ratios,
    classname_start_idx: int = 0,
    include_total: bool = False,
    line_length: int = 20,
) -> List[str]:
    table = []
    split_len = len(splits)
    for j, cls in enumerate(classes):
        info = f"{cls[classname_start_idx:]:<20} |"
        if split_len == 3:
            info += "".join(
                [
                    f" {counts[0][cls]:<5} {'(' + str(round(ratios[0][cls],3)) + ')':<9}",
                    f" {counts[1][cls]:<5} {'(' + str(round(ratios[1][cls],3)) + ')':<9}",
                    f" {counts[2][cls]:<5} {'(' + str(round(ratios[2][cls],3)) + ')':<9}",
                ]
            )
        else:
            info += "".join(
                [
                    f" {counts[0][cls]:<5} {'(' + str(round(ratios[0][cls],3)) + ')':<9}",
                    f" {counts[1][cls]:<5} {'(' + str(round(ratios[1][cls],3)) + ')':<9}",
                ]
            )
        info += "|"
        table.append(info)
    if include_total:
        table.append(create_header_line(line_length))
        info = f"{'Total':<20} |"
        if split_len == 3:
            total = sum(counts[0].values()) + sum(counts[1].values()) + sum(counts[2].values())
            info += "".join(
                [
                    f" {sum(counts[0].values()):<5} {'(' + str(round(sum(counts[0].values())/total,3)) + ')':<9}",
                    f" {sum(counts[1].values()):<5} {'(' + str(round(sum(counts[1].values())/total,3)) + ')':<9}",
                    f" {sum(counts[2].values()):<5} {'(' + str(round(sum(counts[2].values())/total,3)) + ')':<9}",
                ]
            )
        else:
            total = sum(counts[0].values()) + sum(counts[1].values())
            info += "".join(
                [
                    f" {sum(counts[0].values()):<5} {'(' + str(round(sum(counts[0].values())/total,3)) + ')':<9}",
                    f" {sum(counts[1].values()):<5} {'(' + str(round(sum(counts[1].values())/total,3)) + ')':<9}",
                ]
            )
        info += "|"
        table.append(info)
    return table


def create_class_level_table(
    classes,
    splits,
    levels,
    counts,
    ratios,
    classname_start_idx: int = 0,
    include_total: bool = False,
    line_length: int = 20,
) -> List[str]:
    table = []
    split_len = len(splits)
    for j, cls in enumerate(classes):
        info = f"{cls[classname_start_idx:]:<20} |"
        for level in levels:
            if split_len == 3:
                info += "".join(
                    [
                        f" {counts[0][cls][level]:<5} {'(' + str(round(ratios[0][cls][level],3)) + ')':<9}",
                        f" {counts[1][cls][level]:<5} {'(' + str(round(ratios[1][cls][level],3)) + ')':<9}",
                        f" {counts[2][cls][level]:<5} {'(' + str(round(ratios[2][cls][level],3)) + ')':<9}",
                    ]
                )
            else:
                info += "".join(
                    [
                        f" {counts[0][cls][level]:<5} {'(' + str(round(ratios[0][cls][level],3)) + ')':<9}",
                        f" {counts[1][cls][level]:<5} {'(' + str(round(ratios[1][cls][level],3)) + ')':<9}",
                    ]
                )
            info += "|"
        table.append(info)
    if include_total:
        table.append(create_header_line(line_length))
        info = f"{'Total':<20} |"
        for i, level in enumerate(levels):
            if split_len == 3:
                total = [
                    [sum([x[d] for x in counts[i].values()]) for d in levels]
                    for i in range(split_len)
                ]

                ratio_0 = round(total[0][i] / sum(total[0]), 3) if sum(total[0]) != 0 else 0
                ratio_1 = round(total[1][i] / sum(total[1]), 3) if sum(total[1]) != 0 else 0
                ratio_2 = round(total[2][i] / sum(total[2]), 3) if sum(total[2]) != 0 else 0

                info += "".join(
                    [
                        f" {sum([x[level] for x in counts[0].values()]):<5} {'(' + str(ratio_0) + ')':<9}",
                        f" {sum([x[level] for x in counts[1].values()]):<5} {'(' + str(ratio_1) + ')':<9}",
                        f" {sum([x[level] for x in counts[2].values()]):<5} {'(' + str(ratio_2) + ')':<9}",
                    ]
                )
            else:
                total = [
                    [sum([x[d] for x in counts[i].values()]) for d in levels]
                    for i in range(split_len)
                ]
                ratio_0 = round(total[0][i] / sum(total[0]), 3) if sum(total[0]) != 0 else 0
                ratio_1 = round(total[1][i] / sum(total[1]), 3) if sum(total[1]) != 0 else 0

                info += "".join(
                    [
                        f" {sum([x[level] for x in counts[0].values()]):<5} {'(' + str(ratio_0) + ')':<9}",
                        f" {sum([x[level] for x in counts[1].values()]):<5} {'(' + str(ratio_1) + ')':<9}",
                    ]
                )

            info += "|"
        table.append(info)
    return table
