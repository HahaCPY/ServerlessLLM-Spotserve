#!/usr/bin/env python3
"""Render the two-workspace, eight-GPU SpotServe Kubernetes manifests."""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TEMPLATE = (
    REPO_ROOT
    / "deploy/k8s/two-workspace-8gpu/cluster.yaml.in"
)
DEFAULT_TRACE = (
    REPO_ROOT
    / "examples/spotserve/spot_trace_k8_two_workspace_churn.jsonl"
)
DNS_LABEL = re.compile(r"^[a-z0-9](?:[-a-z0-9]*[a-z0-9])?$")


def dns_label(value: str, field: str) -> str:
    value = str(value).strip()
    if not value or len(value) > 63 or not DNS_LABEL.fullmatch(value):
        raise ValueError(f"{field} must be a Kubernetes DNS label: {value!r}")
    return value


def immutable_image(value: str, allow_mutable: bool) -> str:
    value = str(value).strip()
    if not value:
        raise ValueError("image must not be empty")
    if not allow_mutable and "@sha256:" not in value:
        raise ValueError(
            "image must use an immutable @sha256 digest; "
            "pass --allow-mutable-image only for a non-formal smoke"
        )
    return value


def render_manifest(template: str, values: dict[str, str]) -> str:
    rendered = template
    for key, value in values.items():
        rendered = rendered.replace(f"__{key}__", value)
    unresolved = sorted(set(re.findall(r"__[A-Z0-9_]+__", rendered)))
    if unresolved:
        raise ValueError(f"unresolved manifest placeholders: {unresolved}")
    return rendered


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-a-namespace", required=True)
    parser.add_argument("--workspace-b-namespace", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--image-digest", required=True)
    parser.add_argument("--model-pvc-a", required=True)
    parser.add_argument("--model-pvc-b", required=True)
    parser.add_argument("--results-pvc-a", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cluster-name", default="spotserve-k8")
    parser.add_argument("--head-service", default="spotserve-ray-head")
    parser.add_argument("--mem-pool-size", default="4GB")
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--trace", type=Path, default=DEFAULT_TRACE)
    parser.add_argument("--allow-mutable-image", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    workspace_a = dns_label(args.workspace_a_namespace, "workspace A namespace")
    workspace_b = dns_label(args.workspace_b_namespace, "workspace B namespace")
    if workspace_a == workspace_b:
        raise ValueError("workspace A and B namespaces must be different")

    values = {
        "WORKSPACE_A_NAMESPACE": workspace_a,
        "WORKSPACE_B_NAMESPACE": workspace_b,
        "IMAGE": immutable_image(args.image, args.allow_mutable_image),
        "IMAGE_DIGEST": str(args.image_digest).strip(),
        "MODEL_PVC_A": dns_label(args.model_pvc_a, "workspace A model PVC"),
        "MODEL_PVC_B": dns_label(args.model_pvc_b, "workspace B model PVC"),
        "RESULTS_PVC_A": dns_label(args.results_pvc_a, "workspace A results PVC"),
        "CLUSTER_NAME": dns_label(args.cluster_name, "cluster name"),
        "HEAD_SERVICE": dns_label(args.head_service, "head service"),
        "MEM_POOL_SIZE": str(args.mem_pool_size).strip(),
    }
    if not values["IMAGE_DIGEST"]:
        raise ValueError("image digest must not be empty")
    if not values["MEM_POOL_SIZE"]:
        raise ValueError("mem pool size must not be empty")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "manifest": output_dir / "spotserve-two-workspace-8gpu.yaml",
        "values": output_dir / "rendered-values.json",
        "trace": output_dir / args.trace.name,
    }
    existing = [path for path in outputs.values() if path.exists()]
    if existing and not args.force:
        raise FileExistsError(
            "refusing to overwrite rendered artifacts without --force: "
            + ", ".join(str(path) for path in existing)
        )

    template = args.template.read_text(encoding="utf-8")
    outputs["manifest"].write_text(
        render_manifest(template, values), encoding="utf-8"
    )
    outputs["values"].write_text(
        json.dumps(
            {
                **values,
                "HEAD_ADDRESS": (
                    f"{values['HEAD_SERVICE']}.{workspace_a}.svc.cluster.local:6379"
                ),
                "WORKSPACE_A_WORKER_IDS": ["0", "1", "2", "3"],
                "WORKSPACE_B_WORKER_IDS": ["4", "5", "6", "7"],
                "EXPECTED_GPU_COUNT": 8,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    shutil.copyfile(args.trace, outputs["trace"])

    print(f"manifest={outputs['manifest']}")
    print(f"values={outputs['values']}")
    print(f"trace={outputs['trace']}")
    print(
        "head_address="
        f"{values['HEAD_SERVICE']}.{workspace_a}.svc.cluster.local:6379"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
