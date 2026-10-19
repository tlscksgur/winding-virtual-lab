import os
import pickle

from SaigeToolkit.checkpoint.checkpoint_handler import CheckpointHandler
from SaigeToolkit.checkpoint.aes_crypto import encrypt, decrypt
from SaigeToolkit.checkpoint import CHECKPOINT_SPEC, CHECKPOINT_LAST_VERSION
from SaigeToolkit.checkpoint.metadata import read_checkpoint_version
from SaigeToolkit.error.error import InvalidModelFileError
from SaigeToolkit.util.pickle import handle_unpickling_error
from SaigeToolkit.util.version import Version, VersionFormatError

DEFAULT_PASSWORD = os.environ.get("WINDING_CHECKPOINT_PASSWORD", "")


class MergedCheckpointHandler(CheckpointHandler):
    """체크포인트 병합 및 로딩을 처리하는 핸들러 클래스

    이 클래스는 두 개의 체크포인트(main, side)를 병합하고, 암호화된 병합 체크포인트를
    로드하는 기능을 제공합니다. CheckpointHandler를 상속받아 기본 체크포인트 기능을
    확장합니다.

    Attributes:
        checkpoint_sections (list): 병합 체크포인트의 섹션 목록 ["main", "side"]

    Methods:
        merge_checkpoints: 두 체크포인트를 병합하여 암호화된 파일로 저장
        load_merged: 병합된 체크포인트를 로드하고 복호화
        load_merged_checkpoint: 병합 체크포인트 파일에서 데이터 로드
        is_valid_merged_checkpoint: 병합 체크포인트 유효성 검증

    Example:
        >>> # 두 체크포인트 병합
        >>> MergedCheckpointHandler.merge_checkpoints(
        ...     main_checkpoint_dict={"checkpoint_path": "main.ckpt"},
        ...     side_checkpoint_dict={"checkpoint_path": "side.ckpt"},
        ...     save_checkpoint_path="merged.ckpt"
        ... )
        >>>
        >>> # 병합된 체크포인트 로드
        >>> main_ckpt, side_ckpt = MergedCheckpointHandler.load_merged("merged.ckpt")
    """

    checkpoint_sections = ["main", "side"]

    @classmethod
    def merge_checkpoints(
        cls,
        main_checkpoint_dict: dict,
        side_checkpoint_dict: dict,
        save_checkpoint_path: str,
        password: str = DEFAULT_PASSWORD,
    ) -> dict:
        main_checkpoint = cls.load(**main_checkpoint_dict)
        side_checkpoint = cls.load(**side_checkpoint_dict)

        merged_checkpoint = {"main": main_checkpoint, "side": side_checkpoint}

        for section in cls.checkpoint_sections:
            for key in cls._keys_encrypt:
                merged_checkpoint[section][key] = encrypt(merged_checkpoint[section][key], password)

        os.makedirs(os.path.dirname(save_checkpoint_path), exist_ok=True)
        weight_position = CHECKPOINT_SPEC[CHECKPOINT_LAST_VERSION]["weight_position"]
        with open(save_checkpoint_path, "wb") as f:
            f.write(CHECKPOINT_LAST_VERSION)
            f.seek(weight_position)
            pickle.dump(merged_checkpoint, f)

        # TODO: metadata 논의가 완료되면 적용 필요.
        # write_metadata(checkpoint_path=save_path, metadata=metadata)

    @classmethod
    def load_merged(cls, checkpoint_path: str, password: str = DEFAULT_PASSWORD) -> tuple[dict, dict]:
        merged_checkpoint = cls.load_merged_checkpoint(checkpoint_path)

        main_checkpoint = merged_checkpoint["main"]
        side_checkpoint = merged_checkpoint["side"]

        for key in cls._keys_encrypt:
            main_checkpoint[key] = decrypt(main_checkpoint[key], password)
            side_checkpoint[key] = decrypt(side_checkpoint[key], password)

        return main_checkpoint, side_checkpoint

    @classmethod
    @handle_unpickling_error
    def load_merged_checkpoint(cls, checkpoint_path: str) -> dict:
        """Open and load merged checkpoint from path

        Args:
            checkpoint_path (str): merged checkpoint path

        Raises:
            InvalidModelFileError: 다음과 같은 경우 에러 레이즈
            1) 잘못된 WEIGHT_POSITION이 사용된 경우
            2) 파일이 pickle파일이 아닌 경우
            3) 열린 checkpoint 내부의 형식이 올바르지 않은 경우

        Returns:
            Dict: (일부 암호화된) checkpoint
        """
        checkpoint_version = read_checkpoint_version(checkpoint_path)
        if checkpoint_version not in CHECKPOINT_SPEC:
            raise InvalidModelFileError
        weight_position = CHECKPOINT_SPEC[checkpoint_version]["weight_position"]

        with open(checkpoint_path, "rb") as f:
            f.seek(weight_position)
            checkpoint = pickle.load(f)

        if not cls.is_valid_merged_checkpoint(checkpoint):
            raise InvalidModelFileError

        return checkpoint

    @classmethod
    def is_valid_merged_checkpoint(cls, checkpoint: dict) -> bool:
        """Checkpoint가 유효한지 검증하고, 유효하면 True, 그렇지 않으면 False 리턴

        Note:
            현재 로직은 다음 세가지만을 체크함.
            1) checkpoint가 python dictionary 인지
            2) checkpoint가 모든 key를 가지고 있는지
            3) "version" key에 해당하는 value가 유효한지
        """
        if not isinstance(checkpoint, dict):
            return False

        for section in cls.checkpoint_sections:
            if section not in checkpoint:
                return False
            cls.is_valid_checkpoint(checkpoint[section])
        try:
            for section in cls.checkpoint_sections:
                _ = Version.from_string(checkpoint[section].get("version"))
        except VersionFormatError:
            return False

        return True

