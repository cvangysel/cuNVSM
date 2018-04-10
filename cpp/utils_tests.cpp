#include "cuNVSM/tests_base.h"

using ::testing::ElementsAre;
using ::testing::ElementsAreArray;
using ::testing::StrictMock;
using ::testing::Return;

TEST(DataUtils, range) {
    EXPECT_THAT(range<float32>(1, 5, 2),
                ElementsAre(1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0));
}

TEST(Base, flatten) {
    std::vector<size_t> flattened;
    flatten({{8, 9, 10}, {5, 7, 2}, {3}}, &flattened);

    EXPECT_THAT(flattened, ElementsAre(8, 9, 10, 5, 7, 2, 3));
}

TEST(Base, is_number) {
    EXPECT_TRUE(is_number("123"));
    EXPECT_TRUE(is_number("aaa1bbb2ccc3d"));
    EXPECT_FALSE(is_number("hello"));
}

int main(int argc, char* argv[]) {
    google::InitGoogleLogging(argv[0]);
    testing::InitGoogleTest(&argc, argv);
    google::ParseCommandLineFlags(&argc, &argv, true);
    return RUN_ALL_TESTS();
}
