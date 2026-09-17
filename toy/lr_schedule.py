"""Consistent learning-rate CLI controls for every toy arm and run_all."""


def add_lr_arguments(parser):
    for role in ('teacher', 'student'):
        parser.add_argument(f'--{role}-lr', type=float, default=None,
                            help='Initial learning rate (default 1e-3).')
        parser.add_argument(f'--{role}-final-lr', type=float, default=None,
                            help='Learning rate after the drop (default 1e-4).')
        parser.add_argument(f'--{role}-lr-drop-step', type=int, default=None,
                            help='Completed updates before the LR drop (default 40000); -1 disables it.')


def apply_lr_overrides(cfg, args, teacher_lr_field='teacher_lr'):
    for role in ('teacher', 'student'):
        for suffix in ('lr', 'final_lr', 'lr_drop_step'):
            name = f'{role}_{suffix}'
            value = getattr(args, name, None)
            if value is None:
                continue
            if suffix == 'lr_drop_step':
                if value < -1:
                    raise ValueError('LR drop step must be -1 or nonnegative')
                value = None if value == -1 else value
            field = teacher_lr_field if name == 'teacher_lr' else name
            setattr(cfg, field, value)
