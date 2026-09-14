## Description
Briefly describe the change, its motivation, and what problem it solves.

## Type of Change
- [ ] `feat`: New feature or capability
- [ ] `fix`: Bug fix
- [ ] `refactor`: Code refactoring without behavior change
- [ ] `test`: New or updated tests
- [ ] `ci`: CI/CD workflow updates
- [ ] `docs`: Documentation updates
- [ ] `security`: Security fix or hardening
- [ ] `chore`: Maintenance or dependency updates

## Definition of Done Checklist
- [ ] Implementation complete and locally verified
- [ ] All existing tests pass (`pytest`)
- [ ] New tests added for new behavior (if applicable)
- [ ] Code formatted with Black (`black train.py api/ tests/`)
- [ ] No hardcoded secrets, credentials, or magic constants
- [ ] Configuration centralized in `configs/` where appropriate
- [ ] Documentation updated to reflect changes

### Machine Learning Changes (if applicable)
- [ ] Random seed fixed and deterministic
- [ ] Model evaluation metrics recorded and compared against baseline
- [ ] Trusted types verified for Skops serialization
- [ ] Quality gate thresholds satisfied
