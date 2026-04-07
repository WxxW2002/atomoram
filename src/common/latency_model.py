from __future__ import annotations

from dataclasses import dataclass

from src.common.config import ExperimentConfig
from src.common.metrics import AccessMetrics, AccessResult, TimingRecord


@dataclass(slots=True)
class LatencyEstimate:
    online_network_time: float
    offline_network_time: float

    online_server_io_time: float
    offline_server_io_time: float

    online_crypto_time: float
    offline_crypto_time: float

    queueing_delay: float

    online_latency: float
    offline_latency: float
    total_latency: float

    @property
    def total_network_time(self) -> float:
        return self.online_network_time + self.offline_network_time

    @property
    def total_server_io_time(self) -> float:
        return self.online_server_io_time + self.offline_server_io_time

    @property
    def total_crypto_time(self) -> float:
        return self.online_crypto_time + self.offline_crypto_time


class LatencyModel:
    """
    V-ORAM-style modeled latency with an explicit server bucket-I/O term.

    Online latency:
        queueing_delay
      + online communication time
      + online RTT time
      + online client crypto time
      + online server bucket I/O time

    Offline latency:
        offline communication time
      + offline RTT time
      + offline client crypto time
      + offline server bucket I/O time

    Total latency:
        online latency + offline latency
    """

    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        self._validate_config()

    def _validate_config(self) -> None:
        if self.config.network.down_bw_bytes_per_sec <= 0:
            raise ValueError("down_bw_bytes_per_sec must be positive.")
        if self.config.network.up_bw_bytes_per_sec <= 0:
            raise ValueError("up_bw_bytes_per_sec must be positive.")
        if self.config.network.rtt_sec < 0:
            raise ValueError("rtt_sec must be non-negative.")

        if self.config.crypto.enc_bytes_per_sec <= 0:
            raise ValueError("enc_bytes_per_sec must be positive.")
        if self.config.crypto.dec_bytes_per_sec <= 0:
            raise ValueError("dec_bytes_per_sec must be positive.")

        if self.config.server_io.bucket_read_sec < 0:
            raise ValueError("bucket_read_sec must be non-negative.")
        if self.config.server_io.bucket_write_sec < 0:
            raise ValueError("bucket_write_sec must be non-negative.")

    def estimate(
        self,
        metrics: AccessMetrics,
        timing: TimingRecord | None = None,
        *,
        queueing_delay: float | None = None,
    ) -> LatencyEstimate:
        q_delay = self._resolve_queueing_delay(timing=timing, queueing_delay=queueing_delay)

        online_network_time = (
            metrics.online_bytes_down / self.config.network.down_bw_bytes_per_sec
            + metrics.online_bytes_up / self.config.network.up_bw_bytes_per_sec
            + metrics.online_rtt * self.config.network.rtt_sec
        )

        offline_network_time = (
            metrics.offline_bytes_down / self.config.network.down_bw_bytes_per_sec
            + metrics.offline_bytes_up / self.config.network.up_bw_bytes_per_sec
            + metrics.offline_rtt * self.config.network.rtt_sec
        )

        online_server_io_time = (
            metrics.online_bucket_reads * self.config.server_io.bucket_read_sec
            + metrics.online_bucket_writes * self.config.server_io.bucket_write_sec
        )

        offline_server_io_time = (
            metrics.offline_bucket_reads * self.config.server_io.bucket_read_sec
            + metrics.offline_bucket_writes * self.config.server_io.bucket_write_sec
        )

        online_crypto_time = (
            metrics.online_bytes_down / self.config.crypto.dec_bytes_per_sec
            + metrics.online_bytes_up / self.config.crypto.enc_bytes_per_sec
        )

        offline_crypto_time = (
            metrics.offline_bytes_down / self.config.crypto.dec_bytes_per_sec
            + metrics.offline_bytes_up / self.config.crypto.enc_bytes_per_sec
        )

        online_latency = (
            q_delay
            + online_network_time
            + online_server_io_time
            + online_crypto_time
        )

        offline_latency = (
            offline_network_time
            + offline_server_io_time
            + offline_crypto_time
        )

        total_latency = online_latency + offline_latency

        return LatencyEstimate(
            online_network_time=online_network_time,
            offline_network_time=offline_network_time,
            online_server_io_time=online_server_io_time,
            offline_server_io_time=offline_server_io_time,
            online_crypto_time=online_crypto_time,
            offline_crypto_time=offline_crypto_time,
            queueing_delay=q_delay,
            online_latency=online_latency,
            offline_latency=offline_latency,
            total_latency=total_latency,
        )

    def annotate(
        self,
        result: AccessResult,
        *,
        queueing_delay: float | None = None,
    ) -> LatencyEstimate:
        estimate = self.estimate(
            metrics=result.metrics,
            timing=result.timing,
            queueing_delay=queueing_delay,
        )

        result.timing.network_time = estimate.total_network_time
        result.timing.server_io_time = estimate.total_server_io_time
        result.timing.crypto_time = estimate.total_crypto_time
        result.timing.client_cpu_time = 0.0
        result.timing.queueing_delay = estimate.queueing_delay
        result.timing.total_modeled_latency = estimate.total_latency

        # 用户可见响应时间：arrival + online_latency
        if result.timing.arrival_time is not None:
            result.timing.response_time = result.timing.arrival_time + estimate.online_latency
            result.timing.end_to_end_latency = estimate.online_latency
        elif result.timing.service_start_time is not None:
            result.timing.response_time = result.timing.service_start_time + (
                estimate.online_latency - estimate.queueing_delay
            )

        return estimate

    @staticmethod
    def _resolve_queueing_delay(
        *,
        timing: TimingRecord | None,
        queueing_delay: float | None,
    ) -> float:
        if queueing_delay is not None:
            if queueing_delay < 0:
                raise ValueError("queueing_delay must be non-negative.")
            return queueing_delay

        if timing is None:
            return 0.0

        if timing.arrival_time is None or timing.service_start_time is None:
            return 0.0

        inferred = timing.service_start_time - timing.arrival_time
        if inferred < 0:
            raise ValueError(
                "service_start_time is earlier than arrival_time, "
                "which yields a negative queueing delay."
            )
        return inferred