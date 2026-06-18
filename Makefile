.DEFAULT_GOAL := default

include site.mk

# Emulator configuration — override via environment variables.
export EMULATOR_IOS           ?= ios-emulator
export EMULATOR_IOS_DEVICE    ?= iPhone\ 17
export EMULATOR_IOS_RUNTIME   ?= iOS26.5

export ADB_BINARY_PATH         ?= ~/Library/Android/sdk/platform-tools
export EMULATOR_ANDROID        ?= android-emulator
export EMULATOR_ANDROID_DEVICE ?= pixel_9
export EMULATOR_ANDROID_ABI    ?= arm64-v8a

# BEGIN: Primary tasks

default: prepare license_check format analyze test coverage doc_site
.PHONY: default

pre_commit: format_check analyze license_check test
.PHONY: pre_commit

cicd: default
.PHONY: cicd

# END: Primary tasks

# BEGIN: Platform CI targets

# Linux: full quality gate. Run via `make cicd_linux` or the Containerfile.
# doc_site is built separately in the CI workflow to produce the Pages artifact.
cicd_linux: prepare license_check format_check analyze test coverage
.PHONY: cicd_linux

# macOS: Dart unit tests + Flutter integration tests on the macOS device.
cicd_macos: prepare test macos_test
.PHONY: cicd_macos

# Windows: pure Dart tests on Windows.
cicd_windows: prepare test
.PHONY: cicd_windows

# END: Platform CI targets

# BEGIN: Mobile test targets (manual)

# Platform bootstrap — runs flutter create when the directory doesn't exist.
# All three dirs are produced in one pass; declaring them as a multi-target
# means Make will re-run the recipe if any one is missing.
integration_test_app/ios integration_test_app/android integration_test_app/macos: integration_test_app/pubspec.yaml
	@echo "Bootstrapping Flutter platform files for integration_test_app/..."
	cd integration_test_app && \
	flutter create \
	  --template=app \
	  --project-name betto_inferencing_test_app \
	  --org com.bettongia \
	  --platforms=ios,android,macos \
	  .
	# betto_onnxrt_ios requires iOS 16.0+; bump from the flutter create default (13.0).
	sed -i '' \
	  's/IPHONEOS_DEPLOYMENT_TARGET = 13\.0/IPHONEOS_DEPLOYMENT_TARGET = 16.0/g' \
	  integration_test_app/ios/Runner.xcodeproj/project.pbxproj
	# flutter create --template=app always generates test/widget_test.dart; we don't need it.
	rm -f integration_test_app/test/widget_test.dart

# prepare_integration_test_app: explicit alias to pre-bootstrap without running tests.
# The generated android/, ios/, and macos/ directories should be committed.
prepare_integration_test_app: integration_test_app/ios integration_test_app/android integration_test_app/macos
.PHONY: prepare_integration_test_app

# macos_test: Run integration tests on the macOS device.
# Also called by cicd_macos so macOS integration tests run in CI.
macos_test: integration_test_app/macos
	cd integration_test_app && flutter pub get
	cd integration_test_app && flutter test integration_test/ --device-id macos
.PHONY: macos_test

# ios_test: Run integration tests on an iOS simulator.
# Bootstrap and pub get are handled automatically via Make prerequisites.
ios_test: integration_test_app/ios
	cd integration_test_app && flutter pub get
	cd integration_test_app && \
	xcrun simctl list | grep "$(EMULATOR_IOS)" | grep -q "Booted" || xcrun simctl boot $(EMULATOR_IOS) && \
	open -a Simulator && \
	flutter test integration_test/ --device-id $(EMULATOR_IOS)
.PHONY: ios_test

# android_test: Run integration tests on an Android emulator.
# Bootstrap and pub get are handled automatically via Make prerequisites.
android_test: integration_test_app/android
	cd integration_test_app && flutter pub get
	cd integration_test_app && \
	flutter emulators --launch $(EMULATOR_ANDROID) || true && \
	$(ADB_BINARY_PATH)/adb wait-for-device && \
	flutter test integration_test/ --device-id emulator-5554
.PHONY: android_test

# END: Mobile test targets

# BEGIN: Mobile emulator management

emulator_ios_create:
	xcrun simctl create $(EMULATOR_IOS) $(EMULATOR_IOS_DEVICE) $(EMULATOR_IOS_RUNTIME)
.PHONY: emulator_ios_create

emulator_android_create:
	avdmanager create avd \
	  --name $(EMULATOR_ANDROID_DEVICE) \
	  --package "system-images;android-35;google_apis;$(EMULATOR_ANDROID_ABI)" \
	  --device "pixel_9" \
	  --force
.PHONY: emulator_android_create

emulators_stop: emulators_stop_ios emulators_stop_android
.PHONY: emulators_stop

emulators_stop_ios:
	xcrun simctl shutdown $(EMULATOR_IOS) || true
.PHONY: emulators_stop_ios

emulators_stop_android:
	$(ADB_BINARY_PATH)/adb emu kill || true
.PHONY: emulators_stop_android

# END: Mobile emulator management

# container_test: Build and run the CI Linux image locally (requires podman).
container_test:
	podman build -t betto-inferencing-cicd .
	podman run --rm betto-inferencing-cicd
.PHONY: container_test

format:
	dart format lib/ test/  example/
.PHONY: format

format_check:
	dart format --output=none --set-exit-if-changed lib/ test/ example/
.PHONY: format_check

analyze:
	# flutter analyze
	dart analyze
.PHONY: analyze

checks: coverage.log license_check
.PHONY: checks

test: test.log
.PHONY: test

test.log: lib/** test/**
	dart test  | tee test.log


license_check:
	cat addlicense_config.txt | xargs addlicense --check

license_add:
	cat addlicense_config.txt | xargs addlicense

coverage: coverage.log
.PHONY: coverage

coverage.log: lib/** test/**
	# flutter test --coverage
	dart test --coverage-path=coverage/lcov.info
	rm -rf site/coverage
	mkdir -p site/coverage
	genhtml coverage/lcov.info -o site/coverage


# prepare_dart: Dart-only setup — safe on CI runners that lack Flutter.
# prepare_flutter: Full setup including Flutter project pub-gets.
# prepare: Full local setup (delegates to prepare_flutter).
prepare:
	dart pub global activate coverage
	dart pub get
.PHONY: prepare_dart

clean:
	rm -rf site dist coverage .dart_tool
	rm -f *.log
	dart pub get

	cd integration_test_app && flutter clean
	cd integration_test_app && flutter pub get

.PHONY: clean
